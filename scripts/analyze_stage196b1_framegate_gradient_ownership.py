#!/usr/bin/env python3
"""Paired-seed causal analysis of Stage196-B1 FrameGate gradient ownership."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGE = "Stage196-B1-C"
FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT = "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"
SEEDS = (183, 184, 185)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)
ARMS = {"joint": "baseline", "frame_local_only": "intervention"}
BEST_EPOCHS = {"seed183_joint": 20, "seed183_frame_local_only": 20,
               "seed184_joint": 18, "seed184_frame_local_only": 13,
               "seed185_joint": 20, "seed185_frame_local_only": 20}
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
TAIL = (18, 19, 20)
ROW_COUNT, EPOCH_COUNT, SPLIT_SEED = 720, 20, 174
NATIVE_THRESHOLD, SERIALIZATION_TOLERANCE = 0.5, 1e-6
KNOWN_UNIVERSAL = (24, 25, 72, 108, 276, 324, 359, 479, 503, 635)
STAGE196A_DECISION = "STAGE196A_RECURRENT_LOCAL_CHANNEL_FAILURE"
SUPPORTS = "STAGE196B1C_SUPPORTS_DIRECT_FRAMEGATE_GRADIENT_INTERFERENCE"
DOES_NOT_SUPPORT = "STAGE196B1C_DOES_NOT_SUPPORT_DIRECT_FRAMEGATE_GRADIENT_INTERFERENCE"
MIXED = "STAGE196B1C_MIXED_GRADIENT_OWNERSHIP_EFFECT"
INCOMPLETE = "STAGE196B1C_ANALYSIS_INCOMPLETE"
DECISIONS = (SUPPORTS, DOES_NOT_SUPPORT, MIXED, INCOMPLETE)
NEXT = {SUPPORTS: "STAGE196B2_PROMOTE_FRAME_LOCAL_GRADIENT_OWNERSHIP",
        DOES_NOT_SUPPORT: "STAGE196B2_RETURN_TO_FRAME_REPRESENTATION_HYPOTHESIS",
        MIXED: "STAGE196B2_NO_PROMOTION_TARGETED_CAUSAL_FOLLOWUP",
        INCOMPLETE: "STAGE196B1C_REPAIR_ANALYSIS_INPUTS"}
OUTPUTS = ("stage196b1c_analysis.json", "stage196b1c_report.md",
           "stage196b1c_run_summary.csv", "stage196b1c_paired_seed_deltas.csv",
           "stage196b1c_tail3_persistent_rows.csv",
           "stage196b1c_recurrent_position_effects.csv",
           "stage196b1c_epoch_trajectory.csv", "stage196b1c_contract.csv")
REQUIRED_RUN_FILES = ("training_report.json", "run_provenance.json", "clean_dev_predictions.json",
                      "clean_dev_scalars.jsonl", "selected_checkpoint.pt",
                      "stage191_trajectory_contract.json",
                      "stage191_trajectory_epoch_metrics.jsonl")
TRAJECTORY_KEYS = {"epoch", "dev_position", "source_row_id", "gold_final_label",
                   "predicted_final_label", "final_logits", "final_ce", "frame_logit"}
SET_NAMES = ("stage196a_baseline_recurrent", "stage196a_intervention_recurrent",
             "stage196a_common_recurrent", "stage196a_universal_all_six")
SCHEMA = {
    "stable_row_identifier": "predictions[].id / scalars[].id / trajectory[].source_row_id",
    "certified_stable_position": "trajectory[].dev_position",
    "gold_final_label": "predictions[].gold_final_label / trajectory[].gold_final_label",
    "selected_predicted_final_label": "predictions[].pred_final_label",
    "epoch_predicted_final_label": "trajectory[].predicted_final_label",
    "selected_final_probabilities": "predictions[].final_probs, canonical REFUTE/NOT_ENTITLED/SUPPORT order",
    "epoch_final_logits": "trajectory[].final_logits, canonical REFUTE/NOT_ENTITLED/SUPPORT order",
    "frame_probability": "predictions[].frame_prob / scalars[].frame_prob / sigmoid(trajectory[].frame_logit)",
    "predicate_coverage_probability": "predictions[].predicate_coverage_prob / scalars[].predicate_coverage_prob",
    "sufficiency_probability": "predictions[].sufficiency_prob / scalars[].sufficiency_prob",
    "support_probability_or_logit": "predictions[].final_probs[2] / trajectory[].final_logits[2]",
    "not_entitled_probability_or_logit": "predictions[].final_probs[1] / trajectory[].final_logits[1]",
    "intervention_type": "predictions[].intervention_type",
}
RUN_HEADER = ["run", "seed", "arm", "frame_downstream_gradient_mode", "selected_best_epoch",
              "accuracy", "macro_f1", "support_recall", "refute_recall", "not_entitled_recall",
              "support_precision", "false_not_entitled_count", "false_entitlement_count",
              "polarity_error_count", "prediction_counts", "mean_frame_probability_by_gold_label",
              "mean_frame_probability_by_intervention_type", "mean_frame_probability_by_recurrent_set",
              "persistent_stable_support_negative_count", "stable_support_correct_count",
              "unstable_support_count", "persistent_refute_polarity_error_count",
              "persistent_false_entitlement_count",
              "framegate_failure_count_among_persistent_support_negatives",
              "baseline_defined_stable_correct_support_count", "intervention_preserved_support_count",
              "intervention_changed_to_not_entitled_count", "intervention_changed_to_refute_count",
              "intervention_unstable_control_count", "stable_correct_support_preservation_rate"]
DELTA_HEADER = ["metric", "improvement_direction", "seed183_delta", "seed184_delta",
                "seed185_delta", "mean_delta", "median_delta", "number_positive",
                "number_zero", "number_negative"]
TAIL_HEADER = ["run", "seed", "arm", "stable_row_id", "dev_position", "intervention_type",
               "epoch18_prediction", "epoch19_prediction", "epoch20_prediction",
               "selected_best_epoch", "selected_prediction", "frame_probability", "frame_pass",
               "predicate_coverage_probability", "predicate_pass", "sufficiency_probability",
               "sufficiency_pass", "polarity_margin", "polarity_pass", "entitlement_probability",
               "entitlement_aggregation_pass", "final_composition_pass", "mechanism_bucket"]
RECURRENT_HEADER = ["position_set", "seed", "stable_row_id", "dev_position", "intervention_type",
                    "joint_selected_best_frame_probability",
                    "frame_local_only_selected_best_frame_probability",
                    "paired_selected_frame_probability_delta", "joint_tail3_mean_frame_probability",
                    "frame_local_only_tail3_mean_frame_probability",
                    "paired_tail3_frame_probability_delta", "joint_tail3_prediction_pattern",
                    "frame_local_only_tail3_prediction_pattern", "rescued",
                    "previously_correct_harmed"]
EPOCH_HEADER = ["run", "seed", "arm", "frame_downstream_gradient_mode", "epoch",
                "selected_best_epoch", "is_selected_best_epoch", "accuracy", "macro_f1",
                "support_recall", "false_not_entitled_count", "false_entitlement_count",
                "polarity_error_count", "mean_frame_probability_gold_support",
                "mean_frame_probability_persistent_baseline_positions",
                "mean_frame_probability_stage196a_common_recurrent_positions",
                "persistent_tail_membership_precursor_count",
                "persistent_tail_membership_precursor_definition"]
CONTRACT_HEADER = ["scope", "run", "gate", "required", "observed", "passed", "blocking_reason"]
TRAINING_REPORT_EXPECTED = {
    "configured_split_seed": SPLIT_SEED, "resolved_split_seed": SPLIT_SEED,
    "split_seed_explicit": True, "split_policy": "fixed_explicit_split_seed",
    "backbone": "mamba", "model_name": "state-spaces/mamba-130m-hf",
    "architecture": "v6b_minimal", "device": "cuda", "epochs": EPOCH_COUNT,
    "freeze_encoder": True, "freeze_a_log": True, "shared_encoder_trainable": False,
    "shared_encoder_gradient_fully_isolated": True,
    "shared_encoder_isolation_source": "frozen_runtime_configuration",
    "framegate_gradient_ownership_intervention_changed_encoder_freeze_state": False,
    "frame_direct_loss_active": True, "frame_direct_loss_weight": 1.0,
    "frame_downstream_forward_value_changed": False,
}
TRAJECTORY_CONTRACT_EXPECTED = {
    "observability_mode": "stage196b1_framegate_gradient_ownership",
    "stage191_trajectory_observability_implementation_reused": True,
    "authorized_training_seeds": list(SEEDS), "training_seed_authorized": True,
    "freeze_encoder": True, "freeze_a_log": True, "shared_encoder_trainable": False,
    "shared_encoder_gradient_fully_isolated": True,
    "shared_encoder_isolation_source": "frozen_runtime_configuration",
    "framegate_gradient_ownership_intervention_changed_encoder_freeze_state": False,
    "state_capsule_saving_enabled": False, "expected_state_capsules": 0,
    "compatible_positive_margin_enabled": False, "sidecar_accessed": False,
    "parameter_swa_enabled": False, "training_semantics_changed_by_observability": False,
    "extra_forward_pass_performed_by_observability": False,
}
CROSS_SOURCE_FIELDS = (
    "frame_downstream_gradient_mode", "framegate_nonframe_output_gradient_blocked",
    "freeze_encoder", "freeze_a_log", "shared_encoder_trainable",
    "shared_encoder_gradient_fully_isolated", "shared_encoder_isolation_source",
    "framegate_gradient_ownership_intervention_changed_encoder_freeze_state",
)
SPLIT_PROVENANCE_FIELDS = (
    "configured_split_seed", "resolved_split_seed", "split_seed_explicit", "split_policy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--stage196a-report-json", required=True, type=Path)
    p.add_argument("--current-git-commit", required=True)
    p.add_argument("--stage196b1-runtime-git-commit", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def exact_int(v: Any) -> bool: return type(v) is int
def finite(v: Any) -> bool: return type(v) in (int, float) and math.isfinite(float(v))
def avg(v: Sequence[float]) -> float | None: return math.fsum(v) / len(v) if v else None
def ratio(n: int, d: int) -> float | None: return n / d if d else None
def same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=SERIALIZATION_TOLERANCE, abs_tol=SERIALIZATION_TOLERANCE)
def sigmoid(v: float) -> float:
    if v >= 0:
        z = math.exp(-v); return 1.0 / (1.0 + z)
    z = math.exp(v); return z / (1.0 + z)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f: return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): raise ValueError(f"{path}:{n}: blank JSONL row")
            row = json.loads(line)
            if type(row) is not dict: raise ValueError(f"{path}:{n}: row is not an object")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames: raise ValueError(f"{path}: CSV header absent")
        return list(reader)


def git(repo: Path, argv: list[str]) -> str:
    result = subprocess.run(["git", *argv], cwd=repo, check=False,
                            capture_output=True, shell=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8", errors="strict").strip()


def gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
         required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": passed,
                 "blocking_reason": "" if passed else reason})
    if not passed: raise ValueError(f"{run + ': ' if run else ''}{reason}")


def safe_paths(a: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo = a.repo_root.resolve(); reports = (repo / "reports").resolve()
    root = a.run_root.resolve(); prior = a.stage196a_report_json.resolve()
    output = a.output_dir.resolve()
    if not repo.is_dir() or not reports.is_dir(): raise ValueError("repository/reports absent")
    if root != (reports / "stage196b1_framegate_gradient_ownership_runs").resolve():
        raise ValueError("run root is not the exact frozen Stage196-B1 run root")
    if not root.is_dir() or not prior.is_file(): raise FileNotFoundError("run root or Stage196-A report absent")
    if reports != output.parent and reports not in output.parents:
        raise ValueError("output directory must be below repository reports")
    if output in (repo, reports, root) or output == prior.parent:
        raise ValueError("unsafe/colliding output directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory exists and is nonempty")
    return repo, root, prior, output


def recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if type(value) is dict:
        for k, v in value.items():
            if k == key: found.append(v)
            found.extend(recursive_values(v, key))
    elif type(value) is list:
        for item in value: found.extend(recursive_values(item, key))
    return found


def require_value(value: Any, key: str, expected: Any, context: str) -> None:
    observed = recursive_values(value, key)
    if not observed or any(item != expected for item in observed):
        raise ValueError(f"{context}: {key} must be uniformly {expected!r}; got {observed!r}")


def collect_expected(value: Any, expected: dict[str, Any]) -> tuple[dict[str, list[Any]], list[str]]:
    observed = {key: recursive_values(value, key) for key in expected}
    failures = [key for key, wanted in expected.items()
                if not observed[key] or any(item != wanted for item in observed[key])]
    return observed, failures


def parse_bool(value: str, context: str) -> bool:
    if value not in ("True", "False", "true", "false"):
        raise ValueError(f"{context}: invalid Boolean {value!r}")
    return value.lower() == "true"


def validate_source(repo: Path, analysis_commit: str, training_commit: str,
                    rows: list[dict[str, Any]]) -> None:
    analysis_ok = re.fullmatch(r"[0-9a-f]{40}", analysis_commit or "") is not None
    gate(rows, "source", "", "analysis_runtime_commit_format", "lowercase 40-hex",
         analysis_commit, analysis_ok, "invalid analysis runtime Git commit")
    head = git(repo, ["rev-parse", "HEAD"])
    gate(rows, "source", "", "analysis_runtime_commit_equals_head", analysis_commit, head,
         head == analysis_commit, "analysis runtime Git commit differs from HEAD")
    training_ok = re.fullmatch(r"[0-9a-f]{40}", training_commit or "") is not None
    gate(rows, "source", "", "stage196b1_runtime_commit_format", "lowercase 40-hex",
         training_commit, training_ok, "invalid Stage196-B1 runtime Git commit")
    gate(rows, "source", "", "analysis_and_training_commit_roles_separated",
         "independent roles; equal or different values are valid",
         {"analysis_runtime_git_commit": analysis_commit,
          "stage196b1_runtime_git_commit": training_commit}, True, "")
    gate(rows, "source", "", "framegate_implementation_origin_commit_preserved",
         {"role": "FrameGate gradient-ownership implementation origin",
          "git_commit": FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT},
         {"framegate_implementation_origin_git_commit": FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT,
          "analysis_runtime_git_commit": analysis_commit,
          "stage196b1_runtime_git_commit": training_commit}, True, "")


def validate_stage196a(path: Path, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, set[int]]]:
    report = read_json(path)
    required = {"decision": STAGE196A_DECISION, "runnable": True, "blocking_reasons": [],
                "persistent_row_count": 127, "native_channel_threshold": 0.5}
    observed = {key: report.get(key) for key in required} if type(report) is dict else {}
    gate(gates, "stage196a", "", "report_decision_and_count_closure", required, observed,
         observed == required, "Stage196-A decision/count closure mismatch")
    primary = report.get("primary_estimands", {})
    buckets = primary.get("pooled_mechanism_bucket_counts", {})
    if (buckets.get("MULTI_LOCAL_CHANNEL_FAILURE") != 72
            or buckets.get("FRAME_ONLY_FAILURE") != 55
            or sum(int(v) for v in buckets.values()) != 127):
        raise ValueError("Stage196-A 72 multi-local / 55 frame-only closure mismatch")
    if primary.get("recurrent_persistent_position_count_by_arm") != {"baseline": 22, "intervention": 19}:
        raise ValueError("Stage196-A 22/19 recurrent count closure mismatch")
    cross = report.get("cross_arm_recurrence", {})
    if cross.get("recurring_in_both_arms") != 19 or cross.get("persistent_in_all_six_runs") != 10:
        raise ValueError("Stage196-A 19 common / 10 universal closure mismatch")
    source_path = path.with_name("stage196a_source_closure.csv")
    recurrence_path = path.with_name("stage196a_cross_seed_recurrence.csv")
    if not source_path.is_file() or not recurrence_path.is_file():
        raise FileNotFoundError("Stage196-A source-closure/recurrence companion absent")
    source = read_csv(source_path)
    source_ok = bool(source) and all(parse_bool(row.get("passed", ""), "Stage196-A source")
                                     and row.get("blocking_reason", "") == "" for row in source)
    gate(gates, "stage196a", "", "source_closure", True, source_ok, source_ok,
         "Stage196-A source closure contains a failure")
    recurrence = read_csv(recurrence_path)
    required_columns = {"arm", "dev_position", "recurrent_persistent_within_arm",
                        "universal_persistent_within_arm"}
    if not recurrence or not required_columns.issubset(recurrence[0]):
        raise ValueError("Stage196-A recurrence schema unresolved")
    recurrent = {"baseline": set(), "intervention": set()}
    universal = {"baseline": set(), "intervention": set()}
    seen: set[tuple[str, int]] = set()
    for row in recurrence:
        arm = row["arm"]
        if arm not in recurrent: raise ValueError("invalid Stage196-A recurrence arm")
        try: position = int(row["dev_position"])
        except ValueError as exc: raise ValueError("invalid Stage196-A position") from exc
        if not 0 <= position < ROW_COUNT or (arm, position) in seen:
            raise ValueError("duplicate/out-of-range Stage196-A position")
        seen.add((arm, position))
        is_recurrent = parse_bool(row["recurrent_persistent_within_arm"], "Stage196-A recurrence")
        is_universal = parse_bool(row["universal_persistent_within_arm"], "Stage196-A recurrence")
        if is_universal and not is_recurrent: raise ValueError("universal Stage196-A row is not recurrent")
        if is_recurrent: recurrent[arm].add(position)
        if is_universal: universal[arm].add(position)
    sets = {"stage196a_baseline_recurrent": recurrent["baseline"],
            "stage196a_intervention_recurrent": recurrent["intervention"],
            "stage196a_common_recurrent": recurrent["baseline"] & recurrent["intervention"],
            "stage196a_universal_all_six": universal["baseline"] & universal["intervention"]}
    sizes = {key: len(value) for key, value in sets.items()}
    if sizes != dict(zip(SET_NAMES, (22, 19, 19, 10))):
        raise ValueError("recovered Stage196-A set sizes mismatch")
    if tuple(sorted(sets[SET_NAMES[3]])) != KNOWN_UNIVERSAL:
        raise ValueError("universal-position verification invariant failed")
    gate(gates, "stage196a", "", "recurrent_sets_loaded_not_replaced",
         dict(zip(SET_NAMES, (22, 19, 19, 10))), sizes, True, "")
    return report, sets


def vector3(value: Any, context: str, probability: bool = False) -> list[float]:
    if type(value) is not list or len(value) != 3 or any(not finite(v) for v in value):
        raise ValueError(f"{context}: invalid length-three numeric vector")
    result = [float(v) for v in value]
    if probability and (any(v < 0 or v > 1 for v in result)
                        or not math.isclose(math.fsum(result), 1.0, abs_tol=1e-5)):
        raise ValueError(f"{context}: invalid probability vector")
    return result


def canonical(values: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda index: float(values[index]))]


def stable_id(row: dict[str, Any], context: str) -> str:
    value = row.get("id")
    if value is None or isinstance(value, (dict, list)) or not str(value):
        raise ValueError(f"{context}: explicit original row ID absent/invalid")
    return str(value)


def number(row: dict[str, Any], key: str, context: str, probability: bool = False) -> float:
    value = row.get(key)
    if not finite(value): raise ValueError(f"{context}: {key} absent/nonfinite")
    result = float(value)
    if probability and not 0 <= result <= 1: raise ValueError(f"{context}: {key} outside [0,1]")
    return result


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise ValueError("empty classification population")
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows: confusion[row["gold"]][row["prediction"]] += 1
    recalls, precisions, f1s = {}, {}, []
    for label in LABELS:
        tp = confusion[label][label]; gold_n = sum(confusion[label].values())
        pred_n = sum(confusion[gold][label] for gold in LABELS)
        recalls[label] = ratio(tp, gold_n) or 0.0; precisions[label] = ratio(tp, pred_n) or 0.0
        p, r = precisions[label], recalls[label]
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return {"accuracy": sum(confusion[x][x] for x in LABELS)/len(rows),
            "macro_f1": math.fsum(f1s)/3, "support_recall": recalls["SUPPORT"],
            "refute_recall": recalls["REFUTE"], "not_entitled_recall": recalls["NOT_ENTITLED"],
            "support_precision": precisions["SUPPORT"],
            "false_not_entitled_count": sum(r["gold"] in ("REFUTE", "SUPPORT") and r["prediction"] == "NOT_ENTITLED" for r in rows),
            "false_entitlement_count": sum(r["gold"] == "NOT_ENTITLED" and r["prediction"] in ("REFUTE", "SUPPORT") for r in rows),
            "polarity_error_count": sum((r["gold"], r["prediction"]) in (("REFUTE", "SUPPORT"), ("SUPPORT", "REFUTE")) for r in rows),
            "prediction_counts": {label: sum(r["prediction"] == label for r in rows) for label in LABELS},
            "confusion": confusion}


def resolve_run_runtime_commit(run: str, directory: Path, report: dict[str, Any],
                               contract: dict[str, Any], expected: str,
                               gates: list[dict[str, Any]]) -> str:
    provenance_path = (directory / "run_provenance.json").resolve()
    reported_path = Path(str(report.get("run_provenance_json", ""))).resolve()
    gate(gates, "provenance", run, "run_provenance_path",
         str(provenance_path), str(reported_path),
         reported_path == provenance_path and provenance_path.is_file(),
         "authoritative run provenance path mismatch")
    provenance = read_json(provenance_path)
    source = provenance.get("source_provenance") if type(provenance) is dict else None
    observed = {
        "run_provenance.source_provenance.git_commit": (
            source.get("git_commit") if type(source) is dict else None),
        "trajectory_contract.trainer_source_commit": contract.get("trainer_source_commit"),
    }
    valid = all(type(value) is str and re.fullmatch(r"[0-9a-f]{40}", value)
                for value in observed.values())
    matches = valid and set(observed.values()) == {expected}
    gate(gates, "provenance", run, "stage196b1_runtime_commit_authoritative_fields",
         expected, observed, matches,
         "authoritative Stage196-B1 runtime commit fields are absent, contradictory, or mismatched")
    return observed["run_provenance.source_provenance.git_commit"]


def validate_runtime(run: str, seed: int, mode: str, report: dict[str, Any],
                     contract: dict[str, Any], gates: list[dict[str, Any]]) -> None:
    config = report.get("configuration"); split = report.get("split_seed_contract"); trainer = report.get("runs")
    if type(config) is not dict or type(split) is not dict or type(trainer) is not dict:
        raise ValueError(f"{run}: report containers absent")
    if set(trainer) != {"single"} or type(trainer["single"]) is not dict:
        raise ValueError(f"{run}: exact single-run closure mismatch")
    single = trainer["single"]
    training_expected = dict(TRAINING_REPORT_EXPECTED)
    training_expected.update({"training_seed": seed,
                              "frame_downstream_gradient_mode": mode,
                              "framegate_nonframe_output_gradient_blocked": mode == "frame_local_only"})
    nonsplit_expected = {key: value for key, value in training_expected.items()
                         if key not in SPLIT_PROVENANCE_FIELDS}
    config_expected = {key: value for key, value in nonsplit_expected.items() if key != "epochs"}
    training_observed, training_failures = collect_expected(config, config_expected)
    training_observed["epochs"] = {"runs.single.final_epoch": [single.get("final_epoch")]}
    if single.get("final_epoch") != nonsplit_expected["epochs"]:
        training_failures.append("epochs")
    gate(gates, "provenance", run, "training_report_runtime_provenance",
         nonsplit_expected, training_observed, not training_failures,
         f"training-report runtime provenance mismatch: {training_failures}")

    split_expected = {key: training_expected[key] for key in SPLIT_PROVENANCE_FIELDS}
    split_contract_observed, split_failures = collect_expected(split, split_expected)
    split_config_duplicates = {key: recursive_values(config, key) for key in split_expected}
    split_all_occurrences = {key: recursive_values(report, key) for key in split_expected}
    duplicate_failures = [key for key, wanted in split_expected.items()
                          if any(item != wanted for item in split_config_duplicates[key])]
    occurrence_failures = [key for key, wanted in split_expected.items()
                           if not split_all_occurrences[key]
                           or any(item != wanted for item in split_all_occurrences[key])]
    split_observed = {key: {"split_seed_contract": split_contract_observed[key],
                            "configuration_duplicates": split_config_duplicates[key],
                            "training_report_all_occurrences": split_all_occurrences[key]}
                      for key in split_expected}
    split_ok = not split_failures and not duplicate_failures and not occurrence_failures
    gate(gates, "provenance", run, "training_report_split_provenance",
         split_expected, split_observed, split_ok,
         f"training-report split provenance mismatch: authoritative={split_failures}, "
         f"configuration_duplicates={duplicate_failures}, all_occurrences={occurrence_failures}")

    trajectory_expected = dict(TRAJECTORY_CONTRACT_EXPECTED)
    trajectory_expected.update({"arm": ARMS[mode],
                                "frame_downstream_gradient_mode": mode,
                                "framegate_nonframe_output_gradient_blocked": mode == "frame_local_only"})
    trajectory_observed = {key: contract.get(key) for key in trajectory_expected}
    trajectory_failures = [key for key, wanted in trajectory_expected.items()
                           if key not in contract or contract[key] != wanted]
    gate(gates, "provenance", run, "trajectory_observability_provenance",
         trajectory_expected, trajectory_observed, not trajectory_failures,
         f"trajectory observability provenance mismatch: {trajectory_failures}")

    cross_training = {key: training_observed[key][0] for key in CROSS_SOURCE_FIELDS}
    cross_contract = {key: contract.get(key) for key in CROSS_SOURCE_FIELDS}
    cross_expected = {key: nonsplit_expected[key] for key in CROSS_SOURCE_FIELDS}
    cross_failures = [key for key in CROSS_SOURCE_FIELDS
                      if cross_training[key] != cross_contract[key]
                      or cross_training[key] != cross_expected[key]]
    gate(gates, "provenance", run, "cross_source_gradient_ownership_provenance",
         cross_expected, {"training_report": cross_training,
                          "trajectory_contract": cross_contract}, not cross_failures,
         f"cross-source gradient-ownership provenance mismatch: {cross_failures}")

    if config.get("cuda_seed") != seed: raise ValueError(f"{run}: CUDA seed mismatch")
    devices = recursive_values(config, "device") + recursive_values(config, "actual_torch_device")
    if not devices or any(str(v) != "cuda" and not str(v).startswith("cuda:") for v in devices):
        raise ValueError(f"{run}: CUDA provenance mismatch {devices!r}")
    if single.get("final_epoch") != 20 or single.get("best_epoch") != BEST_EPOCHS[run]:
        raise ValueError(f"{run}: final/selected epoch mismatch")

    if (contract.get("epoch_count") != 20 or contract.get("expected_dev_rows") != 720
            or contract.get("training_seed") != seed or contract.get("split_seed") != 174
            or contract.get("arm") != ARMS[mode]):
        raise ValueError(f"{run}: trajectory identity/cardinality mismatch")
    expected_flags = {"stage191_trajectory_replay_observability": False,
                      "stage191_save_trajectory_state_capsules": False,
                      "stage193_tail3_fresh_seed_observability": False,
                      "stage195_tail3_parameter_swa_causal_test": False,
                      "stage196b1_framegate_gradient_ownership_observability": True}
    if contract.get("enabled_flags") != expected_flags: raise ValueError(f"{run}: enabled flags mismatch")
    if contract.get("external_data_used") is not False:
        raise ValueError(f"{run}: external_data_used mismatch")
    margin = config.get("compatible_positive_margin")
    if type(margin) is not dict or margin.get("enabled") is not False or margin.get("weight") != 0.0 or margin.get("margin_logit") != 0.0:
        raise ValueError(f"{run}: compatible-positive margin not exactly off")
    for key in ("external_data_used_for_training", "external_metrics_used_for_threshold_tuning", "time_swap_used"):
        require_value(report, key, False, run)
    for key in ("stage57_bridge_train_mode", "stage66_bridge_train_mode", "stage75_bridge_train_mode", "stage80a_bridge_train_mode"):
        require_value(report, key, "none", run)
    for key in ("stage57_bridge_train_enabled", "stage66_bridge_train_enabled",
                "stage75_bridge_train_enabled", "stage80a_bridge_train_enabled",
                "combined_bridge_enabled", "external_evaluation_active"):
        require_value(report, key, False, run)


def validate_ledger(run: str, rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    if len(rows) != 20: raise ValueError(f"{run}: ledger must have 20 rows")
    by_epoch = {}
    required = {"epoch", "dev_row_count", "clean_accuracy", "clean_macro_f1", "support_recall",
                "false_entitlement_total", "false_not_entitled_total", "polarity_error_total",
                "normalized_prediction_counts", "prediction_export_path", "prediction_export_sha256",
                "capsule_path", "capsule_file_sha256", "trainable_state_sha256", "buffer_state_sha256"}
    for row in rows:
        epoch = row.get("epoch")
        if not exact_int(epoch) or not 1 <= epoch <= 20 or epoch in by_epoch or not required.issubset(row):
            raise ValueError(f"{run}: invalid ledger row")
        if row["dev_row_count"] != 720 or any(row[k] is not None for k in ("capsule_path", "capsule_file_sha256", "trainable_state_sha256", "buffer_state_sha256")):
            raise ValueError(f"{run}: ledger cardinality/capsule mismatch")
        by_epoch[epoch] = row
    if set(by_epoch) != set(range(1, 21)): raise ValueError(f"{run}: ledger epochs mismatch")
    return by_epoch


def validate_run(root: Path, run: str, expected_runtime_commit: str,
                 gates: list[dict[str, Any]]) -> dict[str, Any]:
    seed, mode = int(run[4:7]), run[8:]; directory = (root / run).resolve()
    required = [directory/name for name in REQUIRED_RUN_FILES] + [directory/f"stage191_dev_predictions_epoch_{e:03d}.jsonl" for e in range(1, 21)]
    missing = [p.name for p in required if not p.is_file()]
    gate(gates, "run", run, "required_artifacts", [], missing, not missing, "required artifacts absent")
    counts = {"trajectory": len(list(directory.glob("stage191_dev_predictions_epoch_*.jsonl"))),
              "capsules": len(list(directory.glob("stage191_trajectory_state_epoch_*.pt"))),
              "swa": len(list(directory.glob("stage195*")))}
    gate(gates, "run", run, "trajectory_capsule_swa_closure",
         {"trajectory": 20, "capsules": 0, "swa": 0}, counts,
         counts == {"trajectory": 20, "capsules": 0, "swa": 0}, "trajectory/capsule/SWA closure mismatch")
    report = read_json(directory/"training_report.json"); contract = read_json(directory/"stage191_trajectory_contract.json")
    runtime_commit = resolve_run_runtime_commit(
        run, directory, report, contract, expected_runtime_commit, gates)
    validate_runtime(run, seed, mode, report, contract, gates)
    ledger = validate_ledger(run, read_jsonl(directory/"stage191_trajectory_epoch_metrics.jsonl"))
    obj = read_json(directory/"clean_dev_predictions.json"); scalars = read_jsonl(directory/"clean_dev_scalars.jsonl")
    if type(obj) is not dict or set(obj) != {"metadata", "predictions"}: raise ValueError(f"{run}: prediction container mismatch")
    metadata, predictions = obj["metadata"], obj["predictions"]
    if (type(metadata) is not dict or type(predictions) is not list or len(predictions) != 720
            or len(scalars) != 720 or metadata.get("seed") != seed
            or metadata.get("best_epoch") != BEST_EPOCHS[run]
            or metadata.get("backbone") != "mamba"
            or metadata.get("model_name") != "state-spaces/mamba-130m-hf"
            or metadata.get("architecture") != "v6b_minimal" or metadata.get("freeze_encoder") is not True):
        raise ValueError(f"{run}: selected export metadata/cardinality mismatch")
    pred_by_id, scalar_by_id = {}, {}
    pred_required = {"id", "pair_id", "intervention_type", "claim", "evidence",
                     "gold_final_label", "pred_final_label", "final_probs", "frame_prob",
                     "predicate_coverage_prob", "sufficiency_prob", "entitlement_prob", "polarity_margin"}
    for index, row in enumerate(predictions):
        if type(row) is not dict or not pred_required.issubset(row): raise ValueError(f"{run}: prediction schema unresolved")
        identifier = stable_id(row, f"{run}:prediction:{index}")
        if identifier in pred_by_id: raise ValueError(f"{run}: duplicate prediction ID")
        probs = vector3(row["final_probs"], f"{run}:{identifier}:probabilities", True)
        if row["gold_final_label"] not in LABELS or row["pred_final_label"] != canonical(probs):
            raise ValueError(f"{run}:{identifier}: selected label/probability mismatch")
        for field in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob", "entitlement_prob"):
            number(row, field, f"{run}:{identifier}", True)
        number(row, "polarity_margin", f"{run}:{identifier}"); pred_by_id[identifier] = row
    scalar_required = {"claim", "evidence", "gold_label", "prediction", "frame_logit",
                       "frame_prob", "predicate_coverage_prob", "sufficiency_prob",
                       "entitlement_prob", "polarity_margin", "score_source"}
    for index, row in enumerate(scalars):
        if type(row) is not dict or not scalar_required.issubset(row): raise ValueError(f"{run}: scalar schema unresolved")
        identifier = stable_id(row, f"{run}:scalar:{index}")
        if identifier in scalar_by_id or row["score_source"] != 'direct output["frame_logit"]':
            raise ValueError(f"{run}: duplicate/non-native scalar row")
        scalar_by_id[identifier] = row
    if set(pred_by_id) != set(scalar_by_id): raise ValueError(f"{run}: prediction/scalar ID mismatch")
    selected = {}
    for identifier, pred in pred_by_id.items():
        scalar = scalar_by_id[identifier]
        if any(pred[k] != scalar[k] for k in ("claim", "evidence")) or pred["gold_final_label"] != scalar["gold_label"] or pred["pred_final_label"] != scalar["prediction"]:
            raise ValueError(f"{run}:{identifier}: selected row semantic mismatch")
        for field in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob", "entitlement_prob", "polarity_margin"):
            if not same(number(pred, field, run), number(scalar, field, run)):
                raise ValueError(f"{run}:{identifier}: duplicate field {field} disagreement")
        if not same(sigmoid(number(scalar, "frame_logit", run)), number(scalar, "frame_prob", run)):
            raise ValueError(f"{run}:{identifier}: frame logit/probability mismatch")
        selected[identifier] = {"id": identifier, "claim": pred["claim"], "evidence": pred["evidence"],
                                "gold": pred["gold_final_label"], "prediction": pred["pred_final_label"],
                                "intervention_type": pred["intervention_type"], "pair_id": pred["pair_id"],
                                "probs": vector3(pred["final_probs"], run, True),
                                **{key: float(pred[key]) for key in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob", "entitlement_prob", "polarity_margin")}}
    epochs, position_to_id = {}, {}
    for epoch in range(1, 21):
        path = directory/f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl"; rows = read_jsonl(path)
        if len(rows) != 720: raise ValueError(f"{run}: epoch {epoch} row count mismatch")
        by_id, positions = {}, set()
        for row in rows:
            if set(row) != TRAJECTORY_KEYS or row["epoch"] != epoch or not exact_int(row["dev_position"]) or not 0 <= row["dev_position"] < 720:
                raise ValueError(f"{run}: epoch {epoch} trajectory schema/position mismatch")
            identifier = str(row["source_row_id"]) if row["source_row_id"] is not None else ""
            if not identifier or identifier in by_id or row["dev_position"] in positions or identifier not in selected:
                raise ValueError(f"{run}: epoch {epoch} ambiguous/duplicate stable identity")
            logits = vector3(row["final_logits"], f"{run}:{epoch}:{identifier}")
            if row["gold_final_label"] != selected[identifier]["gold"] or row["predicted_final_label"] != canonical(logits):
                raise ValueError(f"{run}: epoch {epoch} label/logit mismatch")
            frame_logit = number(row, "frame_logit", run)
            by_id[identifier] = {"id": identifier, "position": row["dev_position"],
                                 "gold": row["gold_final_label"], "prediction": row["predicted_final_label"],
                                 "logits": logits, "frame_logit": frame_logit, "frame_prob": sigmoid(frame_logit)}
            positions.add(row["dev_position"])
        if set(by_id) != set(selected) or positions != set(range(720)): raise ValueError(f"{run}: epoch population mismatch")
        mapping = {row["position"]: identifier for identifier, row in by_id.items()}
        if epoch == 1: position_to_id = mapping
        elif mapping != position_to_id: raise ValueError(f"{run}: row drift across epochs")
        epochs[epoch] = by_id
        recomputed = metrics(list(by_id.values())); metric = ledger[epoch]
        checks = {"clean_accuracy": recomputed["accuracy"], "clean_macro_f1": recomputed["macro_f1"],
                  "support_recall": recomputed["support_recall"],
                  "false_entitlement_total": recomputed["false_entitlement_count"],
                  "false_not_entitled_total": recomputed["false_not_entitled_count"],
                  "polarity_error_total": recomputed["polarity_error_count"],
                  "normalized_prediction_counts": recomputed["prediction_counts"]}
        for key, expected in checks.items():
            observed = metric.get(key)
            if not (same(float(observed), float(expected)) if finite(observed) and finite(expected) else observed == expected):
                raise ValueError(f"{run}: epoch {epoch} ledger {key} mismatch")
        if Path(str(metric["prediction_export_path"])).name != path.name or metric["prediction_export_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{run}: epoch {epoch} path/hash mismatch")
    for identifier, row in selected.items():
        selected_epoch_row = epochs[BEST_EPOCHS[run]][identifier]
        if row["prediction"] != selected_epoch_row["prediction"] or not same(row["frame_prob"], selected_epoch_row["frame_prob"]):
            raise ValueError(f"{run}:{identifier}: selected output/epoch mismatch")
        row["position"] = selected_epoch_row["position"]
    gate(gates, "run", run, "stable_id_alignment", "720 unique IDs across 20 epochs", 720, True, "")
    return {"run": run, "seed": seed, "mode": mode, "arm": ARMS[mode], "selected": selected,
            "epochs": epochs, "position_to_id": position_to_id, "report": report, "ledger": ledger,
            "stage196b1_runtime_git_commit": runtime_commit}


def validate_population(runs: dict[str, dict[str, Any]], gates: list[dict[str, Any]]) -> None:
    ref = runs[RUNS[0]]
    for run in RUNS[1:]:
        cur = runs[run]
        if set(cur["selected"]) != set(ref["selected"]) or cur["position_to_id"] != ref["position_to_id"]:
            raise ValueError(f"{run}: cross-run population/position mismatch")
        for identifier, left in ref["selected"].items():
            right = cur["selected"][identifier]
            if any(left[key] != right[key] for key in ("claim", "evidence", "gold", "intervention_type", "pair_id")):
                raise ValueError(f"{run}:{identifier}: immutable metadata drift")
    gate(gates, "source", "", "cross_run_population", "identical 720 stable rows", 720, True, "")


def pattern(data: dict[str, Any], identifier: str) -> tuple[str, str, str]:
    return tuple(data["epochs"][e][identifier]["prediction"] for e in TAIL)  # type: ignore[return-value]
def persistent_ids(data: dict[str, Any]) -> set[str]:
    return {i for i, r in data["selected"].items() if r["gold"] == "SUPPORT" and pattern(data, i) == ("NOT_ENTITLED",)*3}
def stable_ids(data: dict[str, Any]) -> set[str]:
    return {i for i, r in data["selected"].items() if r["gold"] == "SUPPORT" and pattern(data, i) == ("SUPPORT",)*3}
def position_ids(data: dict[str, Any], positions: Iterable[int]) -> list[str]:
    return [data["position_to_id"][p] for p in sorted(positions)]
def selected_frame(data: dict[str, Any], ids: Iterable[str]) -> float | None:
    return avg([data["selected"][i]["frame_prob"] for i in ids])
def epoch_frame(data: dict[str, Any], epoch: int, ids: Iterable[str]) -> float | None:
    return avg([data["epochs"][epoch][i]["frame_prob"] for i in ids])


def localize(row: dict[str, Any]) -> dict[str, Any]:
    f = row["frame_prob"] >= .5; p = row["predicate_coverage_prob"] >= .5
    s = row["sufficiency_prob"] >= .5; a = row["entitlement_prob"] >= .5
    polarity = row["polarity_margin"] >= 0.0
    failed = [name for name, passed in (("frame", f), ("predicate", p), ("sufficiency", s)) if not passed]
    if len(failed) >= 2: bucket = "MULTI_LOCAL_CHANNEL_FAILURE"
    elif failed == ["frame"]: bucket = "FRAME_ONLY_FAILURE"
    elif failed == ["predicate"]: bucket = "PREDICATE_ONLY_FAILURE"
    elif failed == ["sufficiency"]: bucket = "SUFFICIENCY_ONLY_FAILURE"
    elif not a: bucket = "AGGREGATION_FAILURE"
    elif not polarity: bucket = "POLARITY_ONLY_FAILURE"
    elif row["prediction"] != row["gold"]: bucket = "FINAL_COMPOSITION_FAILURE"
    else: bucket = "UNRESOLVED"
    return {"frame_pass": f, "predicate_pass": p, "sufficiency_pass": s,
            "polarity_pass": polarity, "entitlement_aggregation_pass": a,
            "final_composition_pass": row["prediction"] == row["gold"], "mechanism_bucket": bucket}


def controls(joint: dict[str, Any], intervention: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    population = stable_ids(joint); counts = Counter()
    for identifier in population:
        p = pattern(intervention, identifier)
        if p == ("SUPPORT",)*3: counts["preserved"] += 1
        elif p == ("NOT_ENTITLED",)*3: counts["changed_to_not_entitled"] += 1
        elif p == ("REFUTE",)*3: counts["changed_to_refute"] += 1
        else: counts["unstable"] += 1
    result = {key: counts[key] for key in ("preserved", "changed_to_not_entitled", "changed_to_refute", "unstable")}
    result["preservation_rate"] = ratio(counts["preserved"], len(population))
    if sum(result[key] for key in result if key != "preservation_rate") != len(population):
        raise ValueError("control closure mismatch")
    return population, result


def run_summary(data: dict[str, Any], sets: dict[str, set[int]], persistent: set[str], stable: set[str],
                control_population: set[str], control: dict[str, Any]) -> dict[str, Any]:
    rows = list(data["selected"].values()); result = metrics(rows)
    support_total = sum(row["gold"] == "SUPPORT" for row in rows)
    return {"run": data["run"], "seed": data["seed"], "arm": data["arm"],
            "frame_downstream_gradient_mode": data["mode"], "selected_best_epoch": BEST_EPOCHS[data["run"]],
            **{key: result[key] for key in ("accuracy", "macro_f1", "support_recall", "refute_recall",
                                             "not_entitled_recall", "support_precision", "false_not_entitled_count",
                                             "false_entitlement_count", "polarity_error_count", "prediction_counts")},
            "mean_frame_probability_by_gold_label": {label: avg([r["frame_prob"] for r in rows if r["gold"] == label]) for label in LABELS},
            "mean_frame_probability_by_intervention_type": {kind: avg([r["frame_prob"] for r in rows if str(r["intervention_type"]) == kind]) for kind in sorted({str(r["intervention_type"]) for r in rows})},
            "mean_frame_probability_by_recurrent_set": {name: selected_frame(data, position_ids(data, positions)) for name, positions in sets.items()},
            "persistent_stable_support_negative_count": len(persistent), "stable_support_correct_count": len(stable),
            "unstable_support_count": support_total-len(persistent)-len(stable),
            "persistent_refute_polarity_error_count": sum(r["gold"] == "REFUTE" and pattern(data, i) == ("SUPPORT",)*3 for i, r in data["selected"].items()),
            "persistent_false_entitlement_count": sum(r["gold"] == "NOT_ENTITLED" and all(x in ("REFUTE", "SUPPORT") for x in pattern(data, i)) for i, r in data["selected"].items()),
            "framegate_failure_count_among_persistent_support_negatives": sum(not localize(data["selected"][i])["frame_pass"] for i in persistent),
            "baseline_defined_stable_correct_support_count": len(control_population),
            "intervention_preserved_support_count": control.get("preserved", ""),
            "intervention_changed_to_not_entitled_count": control.get("changed_to_not_entitled", ""),
            "intervention_changed_to_refute_count": control.get("changed_to_refute", ""),
            "intervention_unstable_control_count": control.get("unstable", ""),
            "stable_correct_support_preservation_rate": control.get("preservation_rate", "")}


def tail_rows(runs: dict[str, dict[str, Any]], persistent: dict[str, set[str]]) -> list[dict[str, Any]]:
    output = []
    for run in RUNS:
        data = runs[run]
        for identifier in sorted(persistent[run], key=lambda i: data["selected"][i]["position"]):
            row = data["selected"][identifier]; loc = localize(row); p = pattern(data, identifier)
            output.append({"run": run, "seed": data["seed"], "arm": data["arm"],
                           "stable_row_id": identifier, "dev_position": row["position"],
                           "intervention_type": row["intervention_type"], "epoch18_prediction": p[0],
                           "epoch19_prediction": p[1], "epoch20_prediction": p[2],
                           "selected_best_epoch": BEST_EPOCHS[run], "selected_prediction": row["prediction"],
                           "frame_probability": row["frame_prob"], "frame_pass": loc["frame_pass"],
                           "predicate_coverage_probability": row["predicate_coverage_prob"],
                           "predicate_pass": loc["predicate_pass"], "sufficiency_probability": row["sufficiency_prob"],
                           "sufficiency_pass": loc["sufficiency_pass"], "polarity_margin": row["polarity_margin"],
                           "polarity_pass": loc["polarity_pass"], "entitlement_probability": row["entitlement_prob"],
                           "entitlement_aggregation_pass": loc["entitlement_aggregation_pass"],
                           "final_composition_pass": loc["final_composition_pass"], "mechanism_bucket": loc["mechanism_bucket"]})
    return output


def recurrent_rows(runs: dict[str, dict[str, Any]], sets: dict[str, set[int]]) -> list[dict[str, Any]]:
    output = []
    for set_name in SET_NAMES:
        for seed in SEEDS:
            joint = runs[f"seed{seed}_joint"]; intervention = runs[f"seed{seed}_frame_local_only"]
            for position in sorted(sets[set_name]):
                identifier = joint["position_to_id"][position]
                if intervention["position_to_id"][position] != identifier: raise ValueError("recurrent paired identity mismatch")
                js = joint["selected"][identifier]["frame_prob"]; ins = intervention["selected"][identifier]["frame_prob"]
                jt = avg([joint["epochs"][e][identifier]["frame_prob"] for e in TAIL])
                it = avg([intervention["epochs"][e][identifier]["frame_prob"] for e in TAIL])
                assert jt is not None and it is not None
                jp, ip = pattern(joint, identifier), pattern(intervention, identifier)
                output.append({"position_set": set_name, "seed": seed, "stable_row_id": identifier,
                               "dev_position": position, "intervention_type": joint["selected"][identifier]["intervention_type"],
                               "joint_selected_best_frame_probability": js,
                               "frame_local_only_selected_best_frame_probability": ins,
                               "paired_selected_frame_probability_delta": ins-js,
                               "joint_tail3_mean_frame_probability": jt,
                               "frame_local_only_tail3_mean_frame_probability": it,
                               "paired_tail3_frame_probability_delta": it-jt,
                               "joint_tail3_prediction_pattern": list(jp),
                               "frame_local_only_tail3_prediction_pattern": list(ip),
                               "rescued": jp == ("NOT_ENTITLED",)*3 and ip == ("SUPPORT",)*3,
                               "previously_correct_harmed": jp == ("SUPPORT",)*3 and ip != ("SUPPORT",)*3})
    return output


def epoch_rows(runs: dict[str, dict[str, Any]], sets: dict[str, set[int]], baseline_positions: set[int]) -> list[dict[str, Any]]:
    output = []
    for run in RUNS:
        data = runs[run]; baseline_ids = position_ids(data, baseline_positions)
        common_ids = position_ids(data, sets[SET_NAMES[2]])
        support_ids = [i for i, row in data["selected"].items() if row["gold"] == "SUPPORT"]
        for epoch in range(1, 21):
            result = metrics(list(data["epochs"][epoch].values()))
            if epoch < 18: precursor, definition = None, "not_meaningful_before_epoch_18"
            else:
                available = tuple(e for e in TAIL if e <= epoch)
                precursor = sum(data["selected"][i]["gold"] == "SUPPORT" and all(data["epochs"][e][i]["prediction"] == "NOT_ENTITLED" for e in available) for i in data["selected"])
                definition = f"gold_SUPPORT_and_NOT_ENTITLED_at_available_tail_epochs_{list(available)}"
            output.append({"run": run, "seed": data["seed"], "arm": data["arm"],
                           "frame_downstream_gradient_mode": data["mode"], "epoch": epoch,
                           "selected_best_epoch": BEST_EPOCHS[run], "is_selected_best_epoch": epoch == BEST_EPOCHS[run],
                           **{key: result[key] for key in ("accuracy", "macro_f1", "support_recall", "false_not_entitled_count", "false_entitlement_count", "polarity_error_count")},
                           "mean_frame_probability_gold_support": epoch_frame(data, epoch, support_ids),
                           "mean_frame_probability_persistent_baseline_positions": epoch_frame(data, epoch, baseline_ids),
                           "mean_frame_probability_stage196a_common_recurrent_positions": epoch_frame(data, epoch, common_ids),
                           "persistent_tail_membership_precursor_count": precursor,
                           "persistent_tail_membership_precursor_definition": definition})
    if len(output) != 120: raise ValueError("epoch output row count is not 120")
    return output


def delta_table(summaries: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[int, float]]]:
    accessors = {
        "selected_accuracy": ("higher", lambda r: r["accuracy"]),
        "selected_macro_f1": ("higher", lambda r: r["macro_f1"]),
        "selected_support_recall": ("higher", lambda r: r["support_recall"]),
        "selected_false_not_entitled_count": ("lower; negative delta is improvement", lambda r: r["false_not_entitled_count"]),
        "selected_false_entitlement_count": ("lower; negative delta is improvement", lambda r: r["false_entitlement_count"]),
        "selected_polarity_error_count": ("lower; negative delta is improvement", lambda r: r["polarity_error_count"]),
        "persistent_stable_support_negative_count": ("lower; negative delta is improvement", lambda r: r["persistent_stable_support_negative_count"]),
        "framegate_failure_count_among_persistent_support_negatives": ("lower; negative delta is improvement", lambda r: r["framegate_failure_count_among_persistent_support_negatives"]),
        "stable_correct_support_preservation": ("higher; zero means full preservation", lambda r: r["intervention_preserved_support_count"] if r["arm"] == "intervention" else r["baseline_defined_stable_correct_support_count"]),
        "mean_frame_probability_stage196a_common_recurrent": ("higher", lambda r: r["mean_frame_probability_by_recurrent_set"][SET_NAMES[2]]),
        "mean_frame_probability_stage196a_universal_all_six": ("higher", lambda r: r["mean_frame_probability_by_recurrent_set"][SET_NAMES[3]])}
    output, raw = [], {}
    for metric, (direction, accessor) in accessors.items():
        values = {seed: float(accessor(summaries[f"seed{seed}_frame_local_only"])) - float(accessor(summaries[f"seed{seed}_joint"])) for seed in SEEDS}
        raw[metric] = values; vector = list(values.values())
        output.append({"metric": metric, "improvement_direction": direction,
                       **{f"seed{seed}_delta": values[seed] for seed in SEEDS},
                       "mean_delta": avg(vector), "median_delta": statistics.median(vector),
                       "number_positive": sum(v > 0 for v in vector), "number_zero": sum(v == 0 for v in vector),
                       "number_negative": sum(v < 0 for v in vector)})
    return output, raw


def decide(d: dict[str, dict[int, float]]) -> tuple[str, dict[str, Any]]:
    persistent = d["persistent_stable_support_negative_count"]
    frame_fail = d["framegate_failure_count_among_persistent_support_negatives"]
    frame_prob = d["mean_frame_probability_stage196a_common_recurrent"]
    preserve = d["stable_correct_support_preservation"]
    false_ent = d["selected_false_entitlement_count"]; polarity = d["selected_polarity_error_count"]
    conditions = {
        "persistent_lower_at_least_two_not_higher_any": sum(v < 0 for v in persistent.values()) >= 2 and all(v <= 0 for v in persistent.values()),
        "framegate_failures_lower_at_least_two_not_higher_any": sum(v < 0 for v in frame_fail.values()) >= 2 and all(v <= 0 for v in frame_fail.values()),
        "common_recurrent_frame_probability_higher_at_least_two_not_lower_any": sum(v > 0 for v in frame_prob.values()) >= 2 and all(v >= 0 for v in frame_prob.values()),
        "stable_correct_preservation_not_reduced_in_at_least_two": sum(v >= 0 for v in preserve.values()) >= 2,
        "false_entitlement_not_increased_in_at_least_two": sum(v <= 0 for v in false_ent.values()) >= 2,
        "polarity_error_not_increased_in_at_least_two": sum(v <= 0 for v in polarity.values()) >= 2,
    }
    supports = all(conditions.values())
    conflicts = any(any(v < 0 for v in values.values()) and any(v > 0 for v in values.values())
                    for values in (persistent, frame_fail, frame_prob))
    rescue_without_frame = sum(v < 0 for v in persistent.values()) >= 1 and sum(v > 0 for v in frame_prob.values()) < 2
    safety_degrades = sum(v < 0 for v in frame_fail.values()) >= 2 and (sum(v > 0 for v in false_ent.values()) >= 2 or sum(v > 0 for v in polarity.values()) >= 2)
    selected = d["selected_macro_f1"]
    selected_tail_disagree = ((sum(v > 0 for v in selected.values()) >= 2 and any(v > 0 for v in persistent.values()))
                              or (sum(v < 0 for v in selected.values()) >= 2 and any(v < 0 for v in persistent.values())))
    seed_scores = {seed: sum((d[metric][seed] < 0 if "count" in metric or metric == "stable_correct_support_preservation" else d[metric][seed] > 0)
                             for metric in ("persistent_stable_support_negative_count",
                                            "framegate_failure_count_among_persistent_support_negatives",
                                            "mean_frame_probability_stage196a_common_recurrent")) for seed in SEEDS}
    one_seed_dominates = max(seed_scores.values()) == 3 and sum(score >= 2 for score in seed_scores.values()) == 1
    mixed_evidence = {"material_seed_direction_conflict": conflicts,
                      "persistent_rescue_without_coherent_recurrent_frame_probability": rescue_without_frame,
                      "framegate_failure_reduction_with_multiseed_safety_degradation": safety_degrades,
                      "selected_checkpoint_and_tail_three_disagree": selected_tail_disagree,
                      "one_seed_dominates_primary_signature": one_seed_dominates}
    decision = SUPPORTS if supports else MIXED if any(mixed_evidence.values()) else DOES_NOT_SUPPORT
    return decision, {"supports_requirements": conditions, "mixed_conditions": mixed_evidence,
                      "selected_decision": decision}


def analyze(a: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    repo, root, prior_path, _ = safe_paths(a)
    validate_source(repo, a.current_git_commit, a.stage196b1_runtime_git_commit, gates)
    entries = [p.name for p in root.iterdir()]
    exact = set(entries) == set(RUNS) and all((root/run).is_dir() for run in RUNS)
    gate(gates, "source", "", "exact_six_run_root", list(RUNS), entries, exact,
         "run root must contain exactly the six frozen run directories")
    prior, sets = validate_stage196a(prior_path, gates)
    runs = {run: validate_run(root, run, a.stage196b1_runtime_git_commit, gates) for run in RUNS}
    runtime_commits = {run: runs[run]["stage196b1_runtime_git_commit"] for run in RUNS}
    gate(gates, "source", "", "stage196b1_runtime_commit_matches_all_runs",
         a.stage196b1_runtime_git_commit, runtime_commits,
         all(value == a.stage196b1_runtime_git_commit for value in runtime_commits.values()),
         "one or more runs differ from the supplied Stage196-B1 runtime commit")
    observed_runtime_commits = sorted(set(runtime_commits.values()))
    gate(gates, "source", "", "stage196b1_runtime_commit_uniform_across_six_runs",
         [a.stage196b1_runtime_git_commit], observed_runtime_commits,
         observed_runtime_commits == [a.stage196b1_runtime_git_commit],
         "Stage196-B1 runtime commits are not uniform across all six runs")
    validate_population(runs, gates)
    persistent = {run: persistent_ids(runs[run]) for run in RUNS}; stable = {run: stable_ids(runs[run]) for run in RUNS}
    controls_by_seed, control_results = {}, {}
    for seed in SEEDS:
        population, result = controls(runs[f"seed{seed}_joint"], runs[f"seed{seed}_frame_local_only"])
        controls_by_seed[seed], control_results[seed] = population, result
    summaries = {}
    for run in RUNS:
        seed = runs[run]["seed"]
        summaries[run] = run_summary(runs[run], sets, persistent[run], stable[run],
                                     controls_by_seed[seed], control_results[seed] if runs[run]["mode"] == "frame_local_only" else {})
    tails = tail_rows(runs, persistent); recurrents = recurrent_rows(runs, sets)
    baseline_positions = {runs[f"seed{seed}_joint"]["selected"][identifier]["position"]
                          for seed in SEEDS for identifier in persistent[f"seed{seed}_joint"]}
    epochs = epoch_rows(runs, sets, baseline_positions)
    delta_rows, deltas = delta_table(summaries); decision, rule = decide(deltas)
    report = {"stage": STAGE, "decision": decision, "runnable": True, "blocking_reasons": [],
              "analysis_runtime_git_commit": a.current_git_commit,
              "stage196b1_runtime_git_commit": a.stage196b1_runtime_git_commit,
              "analysis_runtime_and_training_runtime_commits_are_distinct_roles": True,
              "framegate_implementation_origin_git_commit": FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT,
              "artifact_only_analysis": True, "training_performed": False, "checkpoint_loaded": False,
              "model_loaded": False, "external_evaluation_performed": False,
              "ordered_runs": list(RUNS), "seeds": list(SEEDS), "tail_epochs": list(TAIL),
              "selected_best_epoch_by_run": BEST_EPOCHS, "selected_and_tail3_kept_distinct": True,
              "resolved_schema": SCHEMA, "row_alignment_key": SCHEMA["stable_row_identifier"],
              "stage196a_decision": prior["decision"],
              "stage196a_recurrent_sets": {key: sorted(value) for key, value in sets.items()},
              "run_summaries": [summaries[run] for run in RUNS],
              "paired_seed_deltas": deltas, "decision_rule_evaluation": rule,
              "authorized_causal_claim": ("Under frozen Mamba, direct non-frame gradients through FrameGate outputs interfered with FrameGate-owned trainable parameters." if decision == SUPPORTS else
                                           "Under frozen Mamba, direct FrameGate-output gradient interference was not supported as the primary cause." if decision == DOES_NOT_SUPPORT else
                                           "No new causal promotion is authorized; the gradient-ownership effect is mixed."),
              "prohibited_claims": ["unfrozen encoder behavior", "external/OOD improvement", "production readiness",
                                    "contrastive-loss necessity", "architecture superiority", "complete mechanistic explanation"],
              "recommended_next_stage": NEXT[decision],
              "remaining_uncertainty": ("The smallest unresolved causal question is whether seed-specific persistent rescue and common-position FrameGate probability move together without entitlement or polarity harm." if decision == MIXED else
                                         "The result is limited to frozen Mamba and direct downstream ownership through FrameGate outputs."),
              "output_file_count": 8, "exception": None}
    tables = {"run": [summaries[run] for run in RUNS], "delta": delta_rows, "tail": tails,
              "recurrent": recurrents, "epoch": epochs, "contract": gates}
    return report, tables


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)): return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool): return "true" if value else "false"
    return value


def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO(newline=""); writer = csv.DictWriter(handle, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header): raise ValueError(f"generated CSV schema mismatch: {set(row)^set(header)}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return handle.getvalue()


def markdown(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    if report["decision"] == INCOMPLETE:
        return ("# Stage196-B1-C FrameGate gradient-ownership analysis\n\n"
                f"## Executive decision\n\n`{INCOMPLETE}`\n\n"
                "## Source and provenance closure\n\nAnalysis failed closed. "
                + " ".join(report["blocking_reasons"]) + "\n\n"
                "## Recommended next stage\n\n`STAGE196B1C_REPAIR_ANALYSIS_INPUTS`\n")
    summary = {row["run"]: row for row in tables["run"]}
    selected_lines = ["| Run | Best epoch | Accuracy | Macro-F1 | SUPPORT recall | False NE | False entitlement | Polarity errors |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tail_lines = ["| Run | Persistent SUPPORT→NE | FrameGate failures | Stable SUPPORT | Unstable SUPPORT |",
                  "|---|---:|---:|---:|---:|"]
    for run in RUNS:
        row = summary[run]
        selected_lines.append(f"| `{run}` | {row['selected_best_epoch']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['support_recall']:.6f} | {row['false_not_entitled_count']} | {row['false_entitlement_count']} | {row['polarity_error_count']} |")
        tail_lines.append(f"| `{run}` | {row['persistent_stable_support_negative_count']} | {row['framegate_failure_count_among_persistent_support_negatives']} | {row['stable_support_correct_count']} | {row['unstable_support_count']} |")
    delta_lines = ["| Metric | Seed 183 | Seed 184 | Seed 185 | Mean | Median | + / 0 / − |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
    for row in tables["delta"]:
        delta_lines.append(f"| {row['metric']} | {row['seed183_delta']:.6g} | {row['seed184_delta']:.6g} | {row['seed185_delta']:.6g} | {row['mean_delta']:.6g} | {row['median_delta']:.6g} | {row['number_positive']} / {row['number_zero']} / {row['number_negative']} |")
    controls_lines = ["| Seed | Baseline stable-correct | Preserved | Changed to NE | Changed to REFUTE | Unstable | Preservation rate |",
                      "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        row = summary[f"seed{seed}_frame_local_only"]
        controls_lines.append(f"| {seed} | {row['baseline_defined_stable_correct_support_count']} | {row['intervention_preserved_support_count']} | {row['intervention_changed_to_not_entitled_count']} | {row['intervention_changed_to_refute_count']} | {row['intervention_unstable_control_count']} | {row['stable_correct_support_preservation_rate']:.6f} |")
    sets = report["stage196a_recurrent_sets"]
    rule = report["decision_rule_evaluation"]
    return "\n".join([
        "# Stage196-B1-C FrameGate gradient-ownership analysis", "", "## Executive decision", "",
        f"`{report['decision']}`", "", report["authorized_causal_claim"], "",
        "## Source and provenance closure", "",
        f"All six runs close to training runtime commit `{report['stage196b1_runtime_git_commit']}`, frozen Mamba `state-spaces/mamba-130m-hf`, `v6b_minimal`, CUDA, 20 epochs, split seed 174, frozen encoder/A_log, no shared-encoder gradient path, and no external, bridge, margin, SWA, calibration, threshold search, or state-capsule activity. Analysis commit: `{report['analysis_runtime_git_commit']}`. These are independent provenance roles; equality is not required.", "",
        "Resolved alignment key: `id`, cross-validated against trajectory `source_row_id`; `dev_position` is used only as a certified stable position. Resolved schema is recorded in the JSON report.", "",
        "## Exact six-run matrix", "", *[f"{i}. `{run}`" for i, run in enumerate(RUNS, 1)], "",
        "## Selected-checkpoint metrics", "", *selected_lines, "",
        "Selected-best outputs above remain distinct from the fixed tail-three analysis; best-epoch movement is not decision evidence by itself.", "",
        "## Tail-three persistent SUPPORT failures", "", "Persistent means gold SUPPORT and NOT_ENTITLED at each of epochs 18, 19, and 20.", "", *tail_lines, "",
        "## FrameGate localization", "", "Frozen Stage196-A native channel threshold 0.5 is reused for frame, predicate, sufficiency, and entitlement aggregation. Polarity pass is the exported SUPPORT-facing margin sign (>= 0); no searched threshold is used. Row-level buckets are in `stage196b1c_tail3_persistent_rows.csv`.", "",
        "## Stage196-A recurrent-position effects", "", f"Separate loaded sets: baseline recurrent={len(sets[SET_NAMES[0]])}, intervention recurrent={len(sets[SET_NAMES[1]])}, common recurrent={len(sets[SET_NAMES[2]])}, universal all-six={len(sets[SET_NAMES[3]])}. Selected and tail-three frame effects and rescue/harm flags are in `stage196b1c_recurrent_position_effects.csv`.", "",
        "## Stable-correct SUPPORT preservation", "", *controls_lines, "",
        "## False-entitlement and polarity safety", "", "Selected-checkpoint paired safety deltas are shown below. Negative error-count delta is improvement.", "",
        "## Twenty-epoch trajectory comparison", "", "`stage196b1c_epoch_trajectory.csv` contains exactly 120 rows (3 seeds × 2 arms × 20 epochs), selected-epoch markers, frame trajectories, and tail-membership precursors. No trainer-selected epoch substitutes for epochs 18–20.", "",
        "## Paired-seed direction table", "", *delta_lines, "",
        "## Decision-rule evaluation", "", "Supports requirements: `" + json.dumps(rule["supports_requirements"], sort_keys=True) + "`", "", "Mixed conditions: `" + json.dumps(rule["mixed_conditions"], sort_keys=True) + "`", "",
        "## Authorized causal claim", "", report["authorized_causal_claim"], "",
        "## Prohibited claims", "", *[f"- {claim}" for claim in report["prohibited_claims"]], "",
        "## Remaining uncertainty", "", report["remaining_uncertainty"], "",
        "## Recommended next stage", "", f"`{report['recommended_next_stage']}`", "",
        "This recommendation does not automatically authorize retraining, a new loss, an architecture change, or external evaluation.", ""])


def incomplete_report(a: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {"stage": STAGE, "decision": INCOMPLETE, "runnable": False,
            "blocking_reasons": [f"{type(exc).__name__}: {exc}"],
            "analysis_runtime_git_commit": a.current_git_commit,
            "stage196b1_runtime_git_commit": a.stage196b1_runtime_git_commit,
            "analysis_runtime_and_training_runtime_commits_are_distinct_roles": True,
            "framegate_implementation_origin_git_commit": FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT,
            "artifact_only_analysis": True,
            "training_performed": False, "checkpoint_loaded": False, "model_loaded": False,
            "external_evaluation_performed": False, "ordered_runs": list(RUNS), "seeds": list(SEEDS),
            "tail_epochs": list(TAIL), "selected_best_epoch_by_run": BEST_EPOCHS,
            "selected_and_tail3_kept_distinct": True, "resolved_schema": SCHEMA,
            "stage196a_recurrent_sets": {}, "run_summaries": [], "paired_seed_deltas": {},
            "decision_rule_evaluation": {"selected_decision": INCOMPLETE},
            "authorized_causal_claim": "No scientific claim is authorized from incomplete analysis.",
            "prohibited_claims": ["all scientific inference"], "recommended_next_stage": NEXT[INCOMPLETE],
            "remaining_uncertainty": "Repair the failed source, provenance, schema, alignment, or contract input.",
            "output_file_count": 8,
            "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}


def contents(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {"stage196b1c_analysis.json": json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)+"\n",
            "stage196b1c_report.md": markdown(report, tables),
            "stage196b1c_run_summary.csv": render_csv(RUN_HEADER, tables["run"]),
            "stage196b1c_paired_seed_deltas.csv": render_csv(DELTA_HEADER, tables["delta"]),
            "stage196b1c_tail3_persistent_rows.csv": render_csv(TAIL_HEADER, tables["tail"]),
            "stage196b1c_recurrent_position_effects.csv": render_csv(RECURRENT_HEADER, tables["recurrent"]),
            "stage196b1c_epoch_trajectory.csv": render_csv(EPOCH_HEADER, tables["epoch"]),
            "stage196b1c_contract.csv": render_csv(CONTRACT_HEADER, tables["contract"])}


def write_exact(output: Path, rendered: dict[str, str]) -> None:
    if set(rendered) != set(OUTPUTS): raise ValueError("internal eight-output closure mismatch")
    output.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        path = output/name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle: handle.write(rendered[name])
    if {path.name for path in output.iterdir()} != set(OUTPUTS):
        raise RuntimeError("written output closure mismatch")


def main() -> int:
    a = parse_args(); gates: list[dict[str, Any]] = []
    empty = {"run": [], "delta": [], "tail": [], "recurrent": [], "epoch": [], "contract": gates}
    try:
        report, tables = analyze(a, gates)
    except Exception as exc:  # fail closed to the precommitted incomplete decision
        gates.append({"scope": "analysis", "run": "", "gate": "analysis_completed",
                      "required": True, "observed": False, "passed": False,
                      "blocking_reason": f"{type(exc).__name__}: {exc}"})
        report, tables = incomplete_report(a, exc), empty
    rendered = contents(report, tables)
    output = a.output_dir.resolve()
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
