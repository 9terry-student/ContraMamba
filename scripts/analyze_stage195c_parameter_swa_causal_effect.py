#!/usr/bin/env python3
"""Analyze the frozen Stage195-B parameter-SWA matrix from artifacts only."""
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
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE195A_RUNTIME_COMMIT = "daddd0eb6f21514ed074f63defa0313323cef555"
TRAINER_BLOB_COMMIT = "bd27e46daf218a57da9a3142c9e4bc5cc44ad53a"
TRAINER_BLOB_SHA256 = "4fe903c9f3aa21ee6365a0297c27e4a333d295dbb851384efc7bc8d3f7607954"
SIDECAR_SHA256 = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
STAGE195A_READY = "STAGE195A_TAIL3_PARAMETER_SWA_MANIFEST_READY"
BLOCKED = "STAGE195C_PARAMETER_SWA_CAUSAL_ANALYSIS_BLOCKED"
REPLICATED_SUPPORT = "STAGE195C_PARAMETER_SWA_REPLICATED_TEMPORAL_CAUSAL_SUPPORT"
BOUNDARY_TRADEOFF = "STAGE195C_PARAMETER_SWA_TEMPORAL_SUPPORT_WITH_BOUNDARY_TRADEOFF"
REPLICATED_HARM = "STAGE195C_PARAMETER_SWA_REPLICATED_CAUSAL_HARM"
NO_SUPPORT = "STAGE195C_PARAMETER_SWA_NO_TEMPORAL_CAUSAL_SUPPORT"
MIXED = "STAGE195C_PARAMETER_SWA_MIXED_OR_INCONCLUSIVE"
DECISIONS = (BLOCKED, REPLICATED_HARM, REPLICATED_SUPPORT, BOUNDARY_TRADEOFF,
             NO_SUPPORT, MIXED)
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
SEEDS = (180, 181, 182)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
EPOCHS = tuple(range(1, 21))
SOURCE_EPOCHS = (18, 19, 20)
ROW_COUNT = 720
SUPPORT_ROWS = 89

STAGE195A_OUTPUTS = {
    "stage195a_tail3_parameter_swa_manifest.json",
    "stage195a_tail3_parameter_swa_manifest.md",
    "stage195a_run_manifest.jsonl",
    "stage195a_run_command_matrix.csv",
    "stage195a_source_and_template_gate.csv",
    "stage195a_precommitted_gate.csv",
}
OUTPUTS = {
    "json": "stage195c_parameter_swa_causal_report.json",
    "md": "stage195c_parameter_swa_causal_report.md",
    "runs": "stage195c_run_summary.csv",
    "rows": "stage195c_row_transition.jsonl",
    "outliers": "stage195c_temporal_outlier_transition.csv",
    "support": "stage195c_support_mechanism_summary.csv",
    "pairs": "stage195c_paired_seed_arm_delta.csv",
    "closure": "stage195c_source_closure.csv",
    "decision": "stage195c_precommitted_decision_gate.csv",
}
GATE_HEADER = ["scope", "run", "gate", "required", "observed", "passed", "blocking_reason"]
DECISION_HEADER = ["decision", "taxonomy_condition", "required", "observed", "passed"]
RUN_HEADER = [
    "run", "seed", "arm", "row_count", "temporal_outlier_count",
    "consensus_correct_outlier_count", "epoch20_correct_outlier_count",
    "both_wrong_outlier_count", "swa_rescue_count", "swa_harm_count",
    "swa_wrong_label_change_count", "swa_unchanged_count",
    "temporal_outlier_net_correctness_change", "epoch20_clean_ce", "swa_clean_ce",
    "clean_ce_delta", "epoch20_accuracy", "swa_accuracy", "accuracy_delta",
    "epoch20_macro_f1", "swa_macro_f1", "macro_f1_delta",
    "epoch20_support_recall", "swa_support_recall", "support_recall_delta",
    "epoch20_false_entitlement_total", "swa_false_entitlement_total",
    "false_entitlement_delta", "epoch20_false_not_entitled_total",
    "swa_false_not_entitled_total", "false_not_entitled_delta",
    "epoch20_polarity_error_total", "swa_polarity_error_total", "polarity_error_delta",
    "epoch20_pred_counts", "swa_pred_counts", "total_corrected_rows",
    "total_harmed_rows", "net_correctness_change",
    "correct_not_entitled_to_false_entitlement",
    "false_entitlement_to_correct_not_entitled",
    "epoch20_SUPPORT_to_swa_NOT_ENTITLED", "epoch20_NOT_ENTITLED_to_swa_SUPPORT",
    "epoch20_REFUTE_to_swa_NOT_ENTITLED", "epoch20_NOT_ENTITLED_to_swa_REFUTE",
    "swa_epoch20_agreement_rate", "swa_mean_logit_agreement_rate",
    "swa_majority_agreement_rate", "swa_majority_temporal_outlier_agreement_rate",
    "swa_mean_temporal_outlier_agreement_rate", "swa_and_mean_both_correct",
    "only_swa_correct", "only_mean_logit_correct", "swa_and_mean_both_wrong",
    "swa_vs_epoch20_l1_mean", "swa_vs_epoch20_l1_median",
    "swa_vs_epoch20_l2_mean", "swa_vs_epoch20_l2_median",
    "swa_vs_mean_logit_l1_mean", "swa_vs_mean_logit_l1_median",
    "swa_vs_mean_logit_l2_mean", "swa_vs_mean_logit_l2_median",
]
OUTLIER_HEADER = [
    "run", "seed", "arm", "temporal_outlier_count", "consensus_correct_count",
    "epoch20_correct_count", "both_wrong_count", "swa_rescue_count", "swa_rescue_rate",
    "swa_harm_count", "swa_harm_rate", "swa_wrong_label_change_count",
    "swa_unchanged_count", "swa_aligns_majority_count", "swa_aligns_majority_rate",
    "swa_aligns_mean_logit_count", "swa_aligns_mean_logit_rate", "net_correctness_change",
]
SUPPORT_HEADER = [
    "run", "seed", "arm", "target_support_consensus_outlier_count",
    "target_swa_support_rescue_count", "target_swa_support_rescue_rate",
    "target_swa_remains_not_entitled_count", "target_swa_remains_not_entitled_rate",
    "target_swa_other_label_count", "target_swa_other_label_rate",
    "target_mean_logit_support_count", "target_mean_logit_support_rate",
    "persistent_stable_support_negative_count", "persistent_swa_support_rescue_count",
    "persistent_swa_support_rescue_rate", "persistent_swa_remains_not_entitled_count",
    "persistent_swa_remains_not_entitled_rate",
    "persistent_mean_logit_remains_not_entitled_count",
    "persistent_mean_logit_remains_not_entitled_rate",
]
PAIR_HEADER = ["seed", "metric", "baseline", "intervention", "intervention_minus_baseline"]
ROW_KEYS = (
    "stage", "run", "seed", "arm", "split_seed", "dev_position", "gold_label",
    "epoch18_prediction", "epoch19_prediction", "epoch20_prediction",
    "mean_logit_prediction", "majority_available", "majority_prediction",
    "swa_prediction", "epoch20_correct", "mean_logit_correct", "majority_correct",
    "swa_correct", "temporal_consensus_outlier", "temporal_outlier_subtype",
    "swa_transition_type", "swa_aligns_majority", "swa_aligns_mean_logit",
    "target_support_consensus_outlier", "persistent_stable_support_negative",
    "epoch18_logits", "epoch19_logits", "epoch20_logits", "tail3_mean_logits",
    "swa_logits", "swa_vs_epoch20_l1", "swa_vs_epoch20_l2",
    "swa_vs_mean_logit_l1", "swa_vs_mean_logit_l2",
)
CSV_HEADERS = {"runs": RUN_HEADER, "outliers": OUTLIER_HEADER,
               "support": SUPPORT_HEADER, "pairs": PAIR_HEADER,
               "closure": GATE_HEADER, "decision": DECISION_HEADER}

STAGE195A_REPORT_KEYS = {
    "stage", "decision", "runnable", "blocking_reasons", "diagnostic_only",
    "exact_six_run_diagnostic_execution_authorized", "model_advancement_authorized",
    "production_swa_selected", "entitlement_correction_implemented",
    "stage195c_decision_made", "subsequent_training_authorized",
    "statistical_significance_claimed", "stage195b_training_performed", "model_loaded",
    "checkpoint_loaded", "external_data_used", "trainer_blob_commit",
    "trainer_blob_sha256", "stage195_runtime_repository_commit", "source_identity",
    "frozen_source_identities", "stage195b_run_root", "ordered_runs",
    "run_manifest_count", "expected_trajectory_rows_per_run",
    "expected_prediction_exports_per_run", "expected_prediction_rows_per_export",
    "expected_stage195_swa_prediction_rows_per_run", "expected_state_capsules_per_run",
    "expected_swa_checkpoints_per_run", "canonical_labels", "logits_source",
    "source_and_template_gates", "precommitted_gates", "exception",
}
STAGE195A_ROW_KEYS = {
    "stage", "run", "training_seed", "split_seed", "arm", "canonical_labels",
    "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
    "stage195_runtime_repository_commit", "argv", "command_argv", "command",
    "planned_run_directory", "planned_output_json_path", "planned_selected_checkpoint_path",
    "expected_trajectory_contract_path", "expected_trajectory_ledger_path",
    "expected_prediction_export_paths", "expected_stage195_swa_predictions_path",
    "expected_stage195_swa_metrics_path", "expected_stage195_swa_contract_path",
    "expected_trajectory_rows", "expected_prediction_exports",
    "expected_prediction_rows_per_export", "expected_stage195_swa_prediction_rows",
    "expected_state_capsules", "expected_swa_checkpoints", "logits_source", "arm_contract",
    "argv_mutation_audit", "frozen_training_envelope", "runnable", "diagnostic_only",
    "exact_six_run_diagnostic_execution_authorized", "model_advancement_authorized",
    "production_swa_selected", "entitlement_correction_implemented", "stage195c_decision_made",
    "subsequent_training_authorized", "statistical_significance_claimed", "external_data_used",
}
MATRIX_HEADER = [
    "run", "training_seed", "split_seed", "arm", "planned_run_directory",
    "planned_output_json_path", "planned_selected_checkpoint_path",
    "expected_trajectory_contract_path", "expected_trajectory_ledger_path",
    "expected_prediction_export_paths", "expected_stage195_swa_predictions_path",
    "expected_stage195_swa_metrics_path", "expected_stage195_swa_contract_path",
    "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
    "stage195_runtime_repository_commit", "command", "expected_trajectory_rows",
    "expected_prediction_exports", "expected_prediction_rows_per_export",
    "expected_stage195_swa_prediction_rows", "expected_state_capsules",
    "expected_swa_checkpoints",
]
SOURCE_GATE_NAMES = (
    "stage195_runtime_and_p0_blob_identity", "stage193a_exact_ready_closure",
    "stage185_semantic_sha256", "baseline_stage193a_template_equivalence",
    "intervention_stage193a_template_equivalence",
    "cross_arm_only_margin_and_sidecar_semantics_differ",
)
PRECOMMITTED_GATE_NAMES = (
    "manifest_strict_non_bool_integer_contract", "exact_stage195b_run_order",
    "planned_artifact_path_and_cardinality_closure", "stage195b_run_root_empty_or_absent",
    "diagnostic_only_authorization",
)
TRAJECTORY_ROW_KEYS = {"epoch", "dev_position", "source_row_id", "gold_final_label",
                       "predicted_final_label", "final_logits", "final_ce", "frame_logit"}
SWA_PREDICTION_KEYS = {"stage", "source", "run", "training_seed", "split_seed", "arm",
                       "source_epochs", "dev_position", "gold_final_label",
                       "predicted_final_label", "final_logits", "final_ce"}
SWA_METRIC_KEYS = {"stage", "source", "run", "training_seed", "split_seed", "arm",
                   "source_epochs", "row_count", "clean_ce", "accuracy", "macro_f1",
                   "support_recall", "false_entitlement_total", "false_not_entitled_total",
                   "polarity_error_total", "pred_counts", "gold_counts", "confusion_matrix",
                   "canonical_labels", "logits_source", "checkpoint_selection_used",
                   "external_data_used"}
SOURCE_FILES = (
    "reports/stage195c_parameter_swa_causal_analysis_spec.md",
    "scripts/analyze_stage195c_parameter_swa_causal_effect.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--stage195a-dir", required=True, type=Path)
    parser.add_argument("--stage195b-run-root", required=True, type=Path)
    parser.add_argument("--current-diagnostic-git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def exact_int(value: Any) -> bool:
    return type(value) is int


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def exact_keys(value: Any, keys: set[str] | frozenset[str], context: str) -> None:
    if type(value) is not dict or set(value) != set(keys):
        observed = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{context}: exact key mismatch: {observed}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL row")
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(row)
    return rows


def read_csv(path: Path, header: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        observed = list(reader.fieldnames or [])
        rows = list(reader)
    if observed != header or any(set(row) != set(header) for row in rows):
        raise ValueError(f"{path}: exact CSV schema mismatch")
    return rows


def bool_csv(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, arguments: list[str], *, binary: bool = False,
        dirty: bool = False) -> Any:
    result = subprocess.run(["git", *arguments], cwd=repo, check=False,
                            capture_output=True, shell=False)
    if dirty:
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def add_gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
             required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": passed,
                 "blocking_reason": "" if passed else reason})
    if not passed:
        raise ValueError(f"{run + ': ' if run else ''}{reason}")


def establish_safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise ValueError("repo root is not a directory")
    reports = (repo / "reports").resolve()
    if not reports.is_dir():
        raise ValueError("reports directory is absent")
    stage195a = args.stage195a_dir.resolve()
    run_root = args.stage195b_run_root.resolve()
    output = args.output_dir.resolve()
    if len({repo, stage195a, run_root, output}) != 4:
        raise ValueError("all supplied paths must be distinct")
    if (stage195a.parent != reports or not stage195a.name.startswith(
            "stage195a_tail3_parameter_swa_manifest_") or not stage195a.is_dir()):
        raise ValueError("Stage195-A directory is unsafe or absent")
    if (run_root.parent != reports or not run_root.name.startswith(
            "stage195b_tail3_parameter_swa_runs_") or not run_root.is_dir()):
        raise ValueError("Stage195-B run root is unsafe or absent")
    if output.parent != reports or not output.name.startswith(
            "stage195c_parameter_swa_causal_analysis_"):
        raise ValueError("Stage195-C output path is unsafe")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Stage195-C output exists and is nonempty")
    return repo, stage195a, run_root, output


def canonical_prediction(logits: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda index: float(logits[index]))]


def mean64(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return math.fsum(float(value) for value in values) / len(values)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def distance(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    return math.fsum(abs(value) for value in differences), math.sqrt(
        math.fsum(value * value for value in differences))


def validate_logits(value: Any, context: str) -> list[float]:
    if type(value) is not list or len(value) != 3 or any(not finite(item) for item in value):
        raise ValueError(f"{context}: invalid length-three finite logits")
    return [float(item) for item in value]


def validate_code_identity(repo: Path, supplied: str,
                           closure: list[dict[str, Any]]) -> dict[str, Any]:
    valid = re.fullmatch(r"[0-9a-f]{40}", supplied or "") is not None
    add_gate(closure, "source", "", "stage195c_commit_format",
             "lowercase 40-hex", supplied, valid, "Stage195-C commit format is invalid")
    head = git(repo, ["rev-parse", "HEAD"])
    add_gate(closure, "source", "", "stage195c_commit_equals_head",
             supplied, head, head == supplied, "Stage195-C commit differs from HEAD")
    identities: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        current = (repo / relative).read_bytes()
        blob = git(repo, ["show", f"{supplied}:{relative}"], binary=True)
        unstaged = git(repo, ["diff", "--quiet", "--", relative], dirty=True) == 0
        staged = git(repo, ["diff", "--cached", "--quiet", "--", relative], dirty=True) == 0
        observed = {"current_sha256": hashlib.sha256(current).hexdigest(),
                    "commit_blob_sha256": hashlib.sha256(blob).hexdigest(),
                    "bytes_equal": current == blob, "unstaged_clean": unstaged,
                    "staged_clean": staged}
        passed = current == blob and unstaged and staged
        add_gate(closure, "source", "", f"stage195c_source_identity:{relative}",
                 {"bytes_equal": True, "unstaged_clean": True, "staged_clean": True},
                 observed, passed, f"Stage195-C source identity failed for {relative}")
        identities[relative] = observed
    return {"stage195c_runtime_repository_commit": supplied, "repository_head": head,
            "files": identities, "passed": True}


def validate_stage195a(stage195a: Path, run_root: Path,
                       closure: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = {path.name for path in stage195a.iterdir()}
    add_gate(closure, "stage195a", "", "exact_six_file_closure", sorted(STAGE195A_OUTPUTS),
             sorted(entries), entries == STAGE195A_OUTPUTS and all(
                 (stage195a / name).is_file() for name in STAGE195A_OUTPUTS),
             "Stage195-A exact six-file closure mismatch")
    report = read_json(stage195a / "stage195a_tail3_parameter_swa_manifest.json")
    exact_keys(report, STAGE195A_REPORT_KEYS, "Stage195-A report")
    required = {"stage": "Stage195-A", "decision": STAGE195A_READY, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True,
        "exact_six_run_diagnostic_execution_authorized": True,
        "model_advancement_authorized": False, "production_swa_selected": False,
        "entitlement_correction_implemented": False, "stage195c_decision_made": False,
        "subsequent_training_authorized": False, "statistical_significance_claimed": False,
        "stage195b_training_performed": False, "model_loaded": False,
        "checkpoint_loaded": False, "external_data_used": False,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_BLOB_SHA256,
        "stage195_runtime_repository_commit": STAGE195A_RUNTIME_COMMIT,
        "stage195b_run_root": str(run_root), "ordered_runs": list(RUNS),
        "run_manifest_count": 6, "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20, "expected_prediction_rows_per_export": 720,
        "expected_stage195_swa_prediction_rows_per_run": 720,
        "expected_state_capsules_per_run": 0, "expected_swa_checkpoints_per_run": 0,
        "canonical_labels": list(LABELS), "logits_source": 'output["logits"]', "exception": None}
    int_fields = {"run_manifest_count": 6, "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20, "expected_prediction_rows_per_export": 720,
        "expected_stage195_swa_prediction_rows_per_run": 720,
        "expected_state_capsules_per_run": 0, "expected_swa_checkpoints_per_run": 0}
    ok = (all(report.get(key) == value for key, value in required.items()) and all(
        exact_int(report.get(key)) and report[key] == value for key, value in int_fields.items()))
    add_gate(closure, "stage195a", "", "ready_report_closure", required,
             {key: report.get(key) for key in required}, ok, "Stage195-A READY closure mismatch")
    for field, names, filename in (("source_and_template_gates", SOURCE_GATE_NAMES,
            "stage195a_source_and_template_gate.csv"), ("precommitted_gates",
            PRECOMMITTED_GATE_NAMES, "stage195a_precommitted_gate.csv")):
        rows = report[field]
        if type(rows) is not list or [row.get("gate") for row in rows] != list(names):
            raise ValueError(f"Stage195-A {field} exact gate order mismatch")
        for row in rows:
            exact_keys(row, {"gate", "required", "observed", "passed", "blocking_reason"}, field)
            if row["passed"] is not True or row["blocking_reason"] != "":
                raise ValueError(f"Stage195-A {field} has a failed gate")
        csv_rows = read_csv(stage195a / filename,
            ["gate", "required", "observed", "passed", "blocking_reason"])
        csv_ok = ([row["gate"] for row in csv_rows] == list(names) and all(
            bool_csv(row["passed"]) is True and row["blocking_reason"] == "" for row in csv_rows))
        add_gate(closure, "stage195a", "", f"all_gates_pass:{filename}", True,
                 [row["gate"] for row in csv_rows], csv_ok, f"Stage195-A gate CSV failed: {filename}")
    manifests = read_jsonl(stage195a / "stage195a_run_manifest.jsonl")
    add_gate(closure, "stage195a", "", "exact_six_run_manifest_order", list(RUNS),
             [row.get("run") for row in manifests],
             len(manifests) == 6 and [row.get("run") for row in manifests] == list(RUNS),
             "Stage195-A manifest order/cardinality mismatch")
    matrix = read_csv(stage195a / "stage195a_run_command_matrix.csv", MATRIX_HEADER)
    add_gate(closure, "stage195a", "", "command_matrix_schema_order", list(RUNS),
             [row["run"] for row in matrix],
             len(matrix) == 6 and [row["run"] for row in matrix] == list(RUNS),
             "Stage195-A command matrix order/cardinality mismatch")
    for manifest, matrix_row, run in zip(manifests, matrix, RUNS):
        exact_keys(manifest, STAGE195A_ROW_KEYS, f"Stage195-A manifest {run}")
        seed, arm, run_dir = int(run[4:7]), run.split("_", 1)[1], (run_root / run).resolve()
        expected_exports = [str((run_dir / f"stage191_dev_predictions_epoch_{e:03d}.jsonl").resolve())
                            for e in EPOCHS]
        required_row = {"stage": "Stage195-A", "run": run, "training_seed": seed,
            "split_seed": 174, "arm": arm, "canonical_labels": list(LABELS),
            "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_BLOB_SHA256,
            "stage195_runtime_repository_commit": STAGE195A_RUNTIME_COMMIT,
            "planned_run_directory": str(run_dir),
            "planned_output_json_path": str((run_dir / "training_report.json").resolve()),
            "expected_trajectory_contract_path": str((run_dir / "stage191_trajectory_contract.json").resolve()),
            "expected_trajectory_ledger_path": str((run_dir / "stage191_trajectory_epoch_metrics.jsonl").resolve()),
            "expected_prediction_export_paths": expected_exports,
            "expected_stage195_swa_predictions_path": str((run_dir / "stage195_tail3_parameter_swa_predictions.jsonl").resolve()),
            "expected_stage195_swa_metrics_path": str((run_dir / "stage195_tail3_parameter_swa_metrics.json").resolve()),
            "expected_stage195_swa_contract_path": str((run_dir / "stage195_tail3_parameter_swa_contract.json").resolve()),
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_stage195_swa_prediction_rows": 720,
            "expected_state_capsules": 0, "expected_swa_checkpoints": 0,
            "logits_source": 'output["logits"]', "runnable": True, "diagnostic_only": True,
            "exact_six_run_diagnostic_execution_authorized": True,
            "model_advancement_authorized": False, "production_swa_selected": False,
            "entitlement_correction_implemented": False, "stage195c_decision_made": False,
            "subsequent_training_authorized": False, "statistical_significance_claimed": False,
            "external_data_used": False}
        if any(manifest.get(key) != value for key, value in required_row.items()):
            raise ValueError(f"{run}: Stage195-A manifest identity mismatch")
        for key in ("training_seed", "split_seed", "expected_trajectory_rows",
                    "expected_prediction_exports", "expected_prediction_rows_per_export",
                    "expected_stage195_swa_prediction_rows", "expected_state_capsules",
                    "expected_swa_checkpoints"):
            if not exact_int(manifest[key]):
                raise ValueError(f"{run}: non-exact integer {key}")
        contract = manifest.get("arm_contract")
        baseline = {"compatible_positive_margin_weight": 0.0,
                    "compatible_positive_margin_logit": 0.0,
                    "controlled_integrity_sidecar_path": None,
                    "expected_integrity_sidecar_semantic_sha256": None}
        arm_ok = contract == baseline if arm == "baseline" else (
            type(contract) is dict and contract.get("compatible_positive_margin_weight") == 0.05
            and contract.get("compatible_positive_margin_logit") == 0.0
            and type(contract.get("controlled_integrity_sidecar_path")) is str
            and contract.get("expected_integrity_sidecar_semantic_sha256") == SIDECAR_SHA256)
        if not arm_ok:
            raise ValueError(f"{run}: arm contract mismatch")
        matrix_required = {"training_seed": str(seed), "split_seed": "174", "arm": arm,
            "planned_run_directory": str(run_dir), "trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "trainer_blob_sha256": TRAINER_BLOB_SHA256,
            "stage195_runtime_repository_commit": STAGE195A_RUNTIME_COMMIT,
            "expected_trajectory_rows": "20", "expected_prediction_exports": "20",
            "expected_prediction_rows_per_export": "720",
            "expected_stage195_swa_prediction_rows": "720", "expected_state_capsules": "0",
            "expected_swa_checkpoints": "0"}
        if any(matrix_row.get(key) != value for key, value in matrix_required.items()):
            raise ValueError(f"{run}: command matrix identity mismatch")
    return report, manifests


def validate_prediction_rows(path: Path, epoch: int, run: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 720:
        raise ValueError(f"{run} epoch {epoch}: expected 720 rows")
    for position, row in enumerate(rows):
        exact_keys(row, TRAJECTORY_ROW_KEYS, f"{run} epoch {epoch} row {position}")
        if (not exact_int(row["epoch"]) or row["epoch"] != epoch
                or not exact_int(row["dev_position"]) or row["dev_position"] != position):
            raise ValueError(f"{run} epoch {epoch}: epoch/position mismatch")
        logits = validate_logits(row["final_logits"], f"{run} epoch {epoch} row {position}")
        if (row["gold_final_label"] not in LABELS or row["predicted_final_label"] not in LABELS
                or row["predicted_final_label"] != canonical_prediction(logits)):
            raise ValueError(f"{run} epoch {epoch}: label/argmax mismatch")
        if not finite(row["final_ce"]) or float(row["final_ce"]) < 0:
            raise ValueError(f"{run} epoch {epoch}: invalid CE")
        if row["frame_logit"] is not None and not finite(row["frame_logit"]):
            raise ValueError(f"{run} epoch {epoch}: invalid frame logit")
    return rows


def validate_swa_rows(path: Path, run: str, seed: int, arm: str,
                      golds: list[str]) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 720:
        raise ValueError(f"{run}: SWA row count mismatch")
    for position, row in enumerate(rows):
        exact_keys(row, SWA_PREDICTION_KEYS, f"{run} SWA row {position}")
        required = {"stage": "Stage195-P0", "source": "tail3_trainable_parameter_swa",
            "run": run, "training_seed": seed, "split_seed": 174, "arm": arm,
            "source_epochs": [18, 19, 20], "dev_position": position,
            "gold_final_label": golds[position]}
        if any(row.get(key) != value for key, value in required.items()) or any(
                not exact_int(row[key]) for key in ("training_seed", "split_seed", "dev_position")):
            raise ValueError(f"{run} SWA row {position}: identity/integer mismatch")
        logits = validate_logits(row["final_logits"], f"{run} SWA row {position}")
        if row["predicted_final_label"] not in LABELS or row[
                "predicted_final_label"] != canonical_prediction(logits):
            raise ValueError(f"{run} SWA row {position}: argmax mismatch")
        if not finite(row["final_ce"]) or float(row["final_ce"]) < 0:
            raise ValueError(f"{run} SWA row {position}: invalid CE")
    return rows


def validate_runs(manifests: list[dict[str, Any]], run_root: Path,
                  closure: list[dict[str, Any]]) -> dict[str, Any]:
    entries = {path.name for path in run_root.iterdir()}
    if entries != set(RUNS) or any(not (run_root / run).is_dir() for run in RUNS):
        raise ValueError("Stage195-B root must contain exactly six run directories")
    data: dict[str, Any] = {}
    global_golds: list[str] | None = None
    for manifest in manifests:
        run, seed, arm = manifest["run"], manifest["training_seed"], manifest["arm"]
        run_dir = (run_root / run).resolve()
        required_names = {"training_report.json", "stage191_trajectory_contract.json",
            "stage191_trajectory_epoch_metrics.jsonl", "stage195_tail3_parameter_swa_predictions.jsonl",
            "stage195_tail3_parameter_swa_metrics.json", "stage195_tail3_parameter_swa_contract.json"}
        required_names.update(f"stage191_dev_predictions_epoch_{e:03d}.jsonl" for e in EPOCHS)
        missing = sorted(name for name in required_names if not (run_dir / name).is_file())
        add_gate(closure, "run", run, "required_artifact_closure", [], missing, not missing,
                 "required artifact is missing")
        capsules = [p.name for p in run_dir.iterdir() if p.is_file() and re.fullmatch(
            r"stage191_trajectory_state_epoch_[0-9]+\.pt", p.name)]
        weights = [p.name for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in
            (".pt", ".pth", ".bin", ".safetensors") and "swa" in p.name.lower()]
        add_gate(closure, "run", run, "zero_capsules_and_swa_weight_artifacts", [],
                 capsules + weights, not capsules and not weights,
                 "state capsule or SWA weight artifact is present")
        contract = read_json(run_dir / "stage191_trajectory_contract.json")
        required_contract = {"observability_mode": "stage195_tail3_parameter_swa_causal_test",
            "authorized_training_seeds": [180, 181, 182], "training_seed_authorized": True,
            "training_seed": seed, "split_seed": 174, "arm": arm, "epoch_count": 20,
            "expected_dev_rows": 720, "expected_gold_support_rows": 89,
            "expected_state_capsules": 0, "source_epochs": [18, 19, 20],
            "canonical_logit_column_labels": list(LABELS), "logits_source": 'output["logits"]',
            "trainer_source_commit": STAGE195A_RUNTIME_COMMIT, "trainer_sha256": TRAINER_BLOB_SHA256,
            "stage191_trajectory_observability_implementation_reused": True,
            "state_capsule_saving_enabled": False, "training_semantics_changed": False,
            "parameter_averaging_changes_training_gradients": False,
            "extra_forward_pass_performed": True, "extra_training_forward_pass_performed": False,
            "post_training_clean_dev_forward_pass_count": 1, "swa_checkpoint_saved": False,
            "calibration_applied": False, "entitlement_boundary_shift_applied": False,
            "loss_logits_used": False, "external_data_used": False}
        contract_ok = all(contract.get(key) == value for key, value in required_contract.items())
        contract_ok = contract_ok and all(exact_int(contract.get(key)) for key in
            ("training_seed", "split_seed", "epoch_count", "expected_dev_rows",
             "expected_gold_support_rows", "expected_state_capsules",
             "post_training_clean_dev_forward_pass_count"))
        add_gate(closure, "run", run, "trajectory_contract", required_contract,
                 {key: contract.get(key) for key in required_contract}, contract_ok,
                 "trajectory contract mismatch")
        ledger = read_jsonl(run_dir / "stage191_trajectory_epoch_metrics.jsonl")
        if len(ledger) != 20 or [row.get("epoch") for row in ledger] != list(EPOCHS) or any(
                not exact_int(row.get("epoch")) for row in ledger):
            raise ValueError(f"{run}: trajectory ledger is not exact epochs 1--20")
        exports = {p.name for p in run_dir.iterdir() if p.is_file() and re.fullmatch(
            r"stage191_dev_predictions_epoch_[0-9]{3}\.jsonl", p.name)}
        expected_exports = {f"stage191_dev_predictions_epoch_{e:03d}.jsonl" for e in EPOCHS}
        if exports != expected_exports:
            raise ValueError(f"{run}: exact twenty prediction export set mismatch")
        retained: dict[int, list[dict[str, Any]]] = {}
        run_golds: list[str] | None = None
        source_ids: list[Any] | None = None
        for epoch, ledger_row in zip(EPOCHS, ledger):
            path = (run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve()
            if (not exact_int(ledger_row.get("dev_row_count")) or ledger_row.get("dev_row_count") != 720
                    or Path(str(ledger_row.get("prediction_export_path", ""))).resolve() != path
                    or ledger_row.get("prediction_export_sha256") != sha256(path)):
                raise ValueError(f"{run} epoch {epoch}: ledger path/hash/cardinality mismatch")
            rows = validate_prediction_rows(path, epoch, run)
            golds, ids = [row["gold_final_label"] for row in rows], [row["source_row_id"] for row in rows]
            if run_golds is None:
                run_golds, source_ids = golds, ids
            if golds != run_golds or ids != source_ids:
                raise ValueError(f"{run} epoch {epoch}: row alignment mismatch")
            if epoch in SOURCE_EPOCHS:
                retained[epoch] = rows
        if run_golds is None or sum(label == "SUPPORT" for label in run_golds) != 89:
            raise ValueError(f"{run}: gold SUPPORT cardinality mismatch")
        if global_golds is None:
            global_golds = run_golds
        if run_golds != global_golds:
            raise ValueError(f"{run}: cross-run gold alignment mismatch")
        swa_path = run_dir / "stage195_tail3_parameter_swa_predictions.jsonl"
        metrics_path = run_dir / "stage195_tail3_parameter_swa_metrics.json"
        swa_rows = validate_swa_rows(swa_path, run, seed, arm, run_golds)
        metrics = read_json(metrics_path)
        exact_keys(metrics, SWA_METRIC_KEYS, f"{run} SWA metrics")
        metric_required = {"stage": "Stage195-P0", "source": "tail3_trainable_parameter_swa",
            "run": run, "training_seed": seed, "split_seed": 174, "arm": arm,
            "source_epochs": [18, 19, 20], "row_count": 720,
            "canonical_labels": list(LABELS), "logits_source": 'output["logits"]',
            "checkpoint_selection_used": False, "external_data_used": False}
        if any(metrics.get(key) != value for key, value in metric_required.items()) or any(
                not exact_int(metrics[key]) for key in ("training_seed", "split_seed", "row_count")):
            raise ValueError(f"{run}: SWA metrics identity mismatch")
        swa_contract = read_json(run_dir / "stage195_tail3_parameter_swa_contract.json")
        contract_required = {"stage": "Stage195-P0", "source": "tail3_trainable_parameter_swa",
            "run": run, "training_seed": seed, "split_seed": 174, "arm": arm,
            "source_epochs": [18, 19, 20], "source_capture_count": 3,
            "accumulator_dtype": "torch.float64", "accumulator_device": "cpu",
            "averaged_values_cast_to_original_dtype": True, "epoch20_restoration_verified": True,
            "post_training_clean_dev_forward_pass_count": 1, "optimizer_state_averaged": False,
            "scheduler_state_averaged": False, "checkpoint_selection_used_for_swa_evaluation": False,
            "swa_checkpoint_saved": False, "calibration_applied": False,
            "entitlement_boundary_shift_applied": False, "external_data_used": False,
            "expected_state_capsules": 0, "expected_prediction_rows": 720,
            "trainer_source_commit": STAGE195A_RUNTIME_COMMIT, "trainer_sha256": TRAINER_BLOB_SHA256}
        if any(swa_contract.get(key) != value for key, value in contract_required.items()) or any(
                not exact_int(swa_contract[key]) for key in ("training_seed", "split_seed",
                    "source_capture_count", "post_training_clean_dev_forward_pass_count",
                    "expected_state_capsules", "expected_prediction_rows")):
            raise ValueError(f"{run}: SWA contract mismatch")
        fingerprint_keys = ("epoch18_trainable_parameter_sha256",
            "epoch19_trainable_parameter_sha256", "epoch20_trainable_parameter_sha256",
            "averaged_trainable_parameter_sha256", "restored_epoch20_trainable_parameter_sha256")
        hashes_ok = (all(re.fullmatch(r"[0-9a-f]{64}", str(swa_contract.get(key, "")))
                         is not None for key in fingerprint_keys)
                     and swa_contract.get("epoch20_trainable_parameter_sha256") ==
                     swa_contract.get("restored_epoch20_trainable_parameter_sha256")
                     and swa_contract.get("predictions_sha256") == sha256(swa_path)
                     and swa_contract.get("metrics_sha256") == sha256(metrics_path)
                     and Path(str(swa_contract.get("predictions_path", ""))).resolve() == swa_path.resolve()
                     and Path(str(swa_contract.get("metrics_path", ""))).resolve() == metrics_path.resolve())
        add_gate(closure, "run", run, "swa_hash_and_restoration_closure", True,
                 {"restoration_equal": swa_contract.get("epoch20_trainable_parameter_sha256") ==
                    swa_contract.get("restored_epoch20_trainable_parameter_sha256"),
                  "prediction_sha256": swa_contract.get("predictions_sha256"),
                  "metrics_sha256": swa_contract.get("metrics_sha256")}, hashes_ok,
                 "SWA hash/path/restoration closure mismatch")
        add_gate(closure, "run", run, "all_twenty_exports_validated", True, True, True, "")
        data[run] = {"seed": seed, "arm": arm, "epochs": retained, "swa": swa_rows,
                     "metrics": metrics}
    return data


def row_ce(logits: Sequence[float], gold: str) -> float:
    maximum = max(float(value) for value in logits)
    logsumexp = maximum + math.log(math.fsum(math.exp(float(value) - maximum) for value in logits))
    return logsumexp - float(logits[LABEL_INDEX[gold]])


def predictor_metrics(rows: list[dict[str, Any]], prediction_key: str,
                      logits_key: str) -> dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    ce_values: list[float] = []
    for row in rows:
        gold, prediction = row["gold_label"], row[prediction_key]
        confusion[gold][prediction] += 1
        ce_values.append(row_ce(row[logits_key], gold))
    pred_counts = {label: sum(confusion[gold][label] for gold in LABELS) for label in LABELS}
    gold_counts = {label: sum(confusion[label].values()) for label in LABELS}
    f1s: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        precision = tp / pred_counts[label] if pred_counts[label] else 0.0
        recall = tp / gold_counts[label] if gold_counts[label] else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    correct = sum(confusion[label][label] for label in LABELS)
    return {"clean_ce": mean64(ce_values), "accuracy": correct / len(rows),
        "macro_f1": mean64(f1s), "support_recall": confusion["SUPPORT"]["SUPPORT"] / 89,
        "false_entitlement_total": (confusion["NOT_ENTITLED"]["REFUTE"]
                                    + confusion["NOT_ENTITLED"]["SUPPORT"]),
        "false_not_entitled_total": (confusion["REFUTE"]["NOT_ENTITLED"]
                                      + confusion["SUPPORT"]["NOT_ENTITLED"]),
        "polarity_error_total": (confusion["REFUTE"]["SUPPORT"]
                                 + confusion["SUPPORT"]["REFUTE"]),
        "pred_counts": pred_counts, "gold_counts": gold_counts, "confusion_matrix": confusion}


def decompose(data: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in RUNS:
        item = data[run]
        for position in range(720):
            late = [item["epochs"][epoch][position] for epoch in SOURCE_EPOCHS]
            predictions = [row["predicted_final_label"] for row in late]
            logits = [validate_logits(row["final_logits"], f"{run} late row {position}")
                      for row in late]
            mean_logits = [mean64([vector[index] for vector in logits]) for index in range(3)]
            counts = Counter(predictions)
            repeated = [label for label in LABELS if counts[label] >= 2]
            majority_available = len(repeated) == 1
            majority = repeated[0] if majority_available else None
            swa_source = item["swa"][position]
            swa_prediction = swa_source["predicted_final_label"]
            swa_logits = validate_logits(swa_source["final_logits"], f"{run} SWA row {position}")
            gold, mean_prediction, epoch20 = late[0]["gold_final_label"], canonical_prediction(mean_logits), predictions[2]
            epoch20_correct, mean_correct = epoch20 == gold, mean_prediction == gold
            majority_correct = majority == gold if majority_available else None
            swa_correct = swa_prediction == gold
            outlier = majority_available and epoch20 != majority
            subtype = ("consensus_correct" if outlier and not epoch20_correct and majority_correct
                       else "epoch20_correct" if outlier and epoch20_correct and not majority_correct
                       else "both_wrong" if outlier else "not_temporal_outlier")
            transition = ("swa_rescue" if outlier and not epoch20_correct and swa_correct
                          else "swa_harm" if outlier and epoch20_correct and not swa_correct
                          else "swa_wrong_label_change" if outlier and not epoch20_correct
                               and not swa_correct and swa_prediction != epoch20
                          else "swa_unchanged")
            l1e, l2e = distance(swa_logits, logits[2])
            l1m, l2m = distance(swa_logits, mean_logits)
            row = {"stage": "Stage195-C", "run": run, "seed": item["seed"],
                "arm": item["arm"], "split_seed": 174, "dev_position": position,
                "gold_label": gold, "epoch18_prediction": predictions[0],
                "epoch19_prediction": predictions[1], "epoch20_prediction": epoch20,
                "mean_logit_prediction": mean_prediction, "majority_available": majority_available,
                "majority_prediction": majority, "swa_prediction": swa_prediction,
                "epoch20_correct": epoch20_correct, "mean_logit_correct": mean_correct,
                "majority_correct": majority_correct, "swa_correct": swa_correct,
                "temporal_consensus_outlier": outlier, "temporal_outlier_subtype": subtype,
                "swa_transition_type": transition,
                "swa_aligns_majority": majority_available and swa_prediction == majority,
                "swa_aligns_mean_logit": swa_prediction == mean_prediction,
                "target_support_consensus_outlier": (gold == "SUPPORT" and epoch20 ==
                    "NOT_ENTITLED" and majority_available and majority == "SUPPORT"),
                "persistent_stable_support_negative": (gold == "SUPPORT" and all(
                    prediction == "NOT_ENTITLED" for prediction in predictions)),
                "epoch18_logits": logits[0], "epoch19_logits": logits[1],
                "epoch20_logits": logits[2], "tail3_mean_logits": mean_logits,
                "swa_logits": swa_logits, "swa_vs_epoch20_l1": l1e,
                "swa_vs_epoch20_l2": l2e, "swa_vs_mean_logit_l1": l1m,
                "swa_vs_mean_logit_l2": l2m}
            if tuple(row) != ROW_KEYS:
                raise RuntimeError("row-transition exact schema drift")
            output.append(row)
    if len(output) != 4320:
        raise RuntimeError("row-transition cardinality is not exactly 4320")
    return output


def summary_for_run(run: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    epoch20 = predictor_metrics(rows, "epoch20_prediction", "epoch20_logits")
    swa = predictor_metrics(rows, "swa_prediction", "swa_logits")
    outliers = [row for row in rows if row["temporal_consensus_outlier"]]
    count = lambda predicate, source=rows: sum(bool(predicate(row)) for row in source)
    rescue = count(lambda row: row["swa_transition_type"] == "swa_rescue", outliers)
    harm = count(lambda row: row["swa_transition_type"] == "swa_harm", outliers)
    majority_rows = [row for row in rows if row["majority_available"]]
    distances = lambda key: [float(row[key]) for row in rows]
    result: dict[str, Any] = {"run": run, "seed": rows[0]["seed"], "arm": rows[0]["arm"],
        "row_count": len(rows), "temporal_outlier_count": len(outliers),
        "consensus_correct_outlier_count": count(lambda r: r["temporal_outlier_subtype"] == "consensus_correct", outliers),
        "epoch20_correct_outlier_count": count(lambda r: r["temporal_outlier_subtype"] == "epoch20_correct", outliers),
        "both_wrong_outlier_count": count(lambda r: r["temporal_outlier_subtype"] == "both_wrong", outliers),
        "swa_rescue_count": rescue, "swa_harm_count": harm,
        "swa_wrong_label_change_count": count(lambda r: r["swa_transition_type"] == "swa_wrong_label_change", outliers),
        "swa_unchanged_count": count(lambda r: r["swa_transition_type"] == "swa_unchanged", outliers),
        "temporal_outlier_net_correctness_change": rescue - harm}
    for metric in ("clean_ce", "accuracy", "macro_f1", "support_recall",
                   "false_entitlement_total", "false_not_entitled_total", "polarity_error_total"):
        result[f"epoch20_{metric}"] = epoch20[metric]
        result[f"swa_{metric}"] = swa[metric]
        result[metric.replace("_total", "") + "_delta"] = swa[metric] - epoch20[metric]
    result.update({"epoch20_pred_counts": epoch20["pred_counts"], "swa_pred_counts": swa["pred_counts"],
        "total_corrected_rows": count(lambda r: not r["epoch20_correct"] and r["swa_correct"]),
        "total_harmed_rows": count(lambda r: r["epoch20_correct"] and not r["swa_correct"]),
        "correct_not_entitled_to_false_entitlement": count(lambda r: r["gold_label"] == "NOT_ENTITLED"
            and r["epoch20_prediction"] == "NOT_ENTITLED" and r["swa_prediction"] != "NOT_ENTITLED"),
        "false_entitlement_to_correct_not_entitled": count(lambda r: r["gold_label"] == "NOT_ENTITLED"
            and r["epoch20_prediction"] != "NOT_ENTITLED" and r["swa_prediction"] == "NOT_ENTITLED")})
    result["net_correctness_change"] = result["total_corrected_rows"] - result["total_harmed_rows"]
    for source, target in (("SUPPORT", "NOT_ENTITLED"), ("NOT_ENTITLED", "SUPPORT"),
                           ("REFUTE", "NOT_ENTITLED"), ("NOT_ENTITLED", "REFUTE")):
        result[f"epoch20_{source}_to_swa_{target}"] = count(lambda r, a=source, b=target:
            r["epoch20_prediction"] == a and r["swa_prediction"] == b)
    result.update({
        "swa_epoch20_agreement_rate": ratio(count(lambda r: r["swa_prediction"] == r["epoch20_prediction"]), len(rows)),
        "swa_mean_logit_agreement_rate": ratio(count(lambda r: r["swa_aligns_mean_logit"]), len(rows)),
        "swa_majority_agreement_rate": ratio(count(lambda r: r["swa_aligns_majority"], majority_rows), len(majority_rows)),
        "swa_majority_temporal_outlier_agreement_rate": ratio(count(lambda r: r["swa_aligns_majority"], outliers), len(outliers)),
        "swa_mean_temporal_outlier_agreement_rate": ratio(count(lambda r: r["swa_aligns_mean_logit"], outliers), len(outliers)),
        "swa_and_mean_both_correct": count(lambda r: r["swa_correct"] and r["mean_logit_correct"]),
        "only_swa_correct": count(lambda r: r["swa_correct"] and not r["mean_logit_correct"]),
        "only_mean_logit_correct": count(lambda r: not r["swa_correct"] and r["mean_logit_correct"]),
        "swa_and_mean_both_wrong": count(lambda r: not r["swa_correct"] and not r["mean_logit_correct"])})
    for key in ("swa_vs_epoch20_l1", "swa_vs_epoch20_l2", "swa_vs_mean_logit_l1", "swa_vs_mean_logit_l2"):
        values = distances(key)
        result[key + "_mean"] = mean64(values)
        result[key + "_median"] = statistics.median(values)
    if set(result) != set(RUN_HEADER):
        raise RuntimeError(f"{run}: run summary schema drift: {sorted(set(RUN_HEADER) ^ set(result))}")
    outlier_row = {"run": run, "seed": result["seed"], "arm": result["arm"],
        "temporal_outlier_count": len(outliers),
        "consensus_correct_count": result["consensus_correct_outlier_count"],
        "epoch20_correct_count": result["epoch20_correct_outlier_count"],
        "both_wrong_count": result["both_wrong_outlier_count"], "swa_rescue_count": rescue,
        "swa_rescue_rate": ratio(rescue, len(outliers)), "swa_harm_count": harm,
        "swa_harm_rate": ratio(harm, len(outliers)),
        "swa_wrong_label_change_count": result["swa_wrong_label_change_count"],
        "swa_unchanged_count": result["swa_unchanged_count"],
        "swa_aligns_majority_count": count(lambda r: r["swa_aligns_majority"], outliers),
        "swa_aligns_majority_rate": ratio(count(lambda r: r["swa_aligns_majority"], outliers), len(outliers)),
        "swa_aligns_mean_logit_count": count(lambda r: r["swa_aligns_mean_logit"], outliers),
        "swa_aligns_mean_logit_rate": ratio(count(lambda r: r["swa_aligns_mean_logit"], outliers), len(outliers)),
        "net_correctness_change": rescue - harm}
    target = [row for row in rows if row["target_support_consensus_outlier"]]
    persistent = [row for row in rows if row["persistent_stable_support_negative"]]
    t_rescue = count(lambda r: r["swa_prediction"] == "SUPPORT", target)
    t_remain = count(lambda r: r["swa_prediction"] == "NOT_ENTITLED", target)
    p_rescue = count(lambda r: r["swa_prediction"] == "SUPPORT", persistent)
    p_remain = count(lambda r: r["swa_prediction"] == "NOT_ENTITLED", persistent)
    support_row = {"run": run, "seed": result["seed"], "arm": result["arm"],
        "target_support_consensus_outlier_count": len(target),
        "target_swa_support_rescue_count": t_rescue, "target_swa_support_rescue_rate": ratio(t_rescue, len(target)),
        "target_swa_remains_not_entitled_count": t_remain, "target_swa_remains_not_entitled_rate": ratio(t_remain, len(target)),
        "target_swa_other_label_count": len(target) - t_rescue - t_remain,
        "target_swa_other_label_rate": ratio(len(target) - t_rescue - t_remain, len(target)),
        "target_mean_logit_support_count": count(lambda r: r["mean_logit_prediction"] == "SUPPORT", target),
        "target_mean_logit_support_rate": ratio(count(lambda r: r["mean_logit_prediction"] == "SUPPORT", target), len(target)),
        "persistent_stable_support_negative_count": len(persistent),
        "persistent_swa_support_rescue_count": p_rescue, "persistent_swa_support_rescue_rate": ratio(p_rescue, len(persistent)),
        "persistent_swa_remains_not_entitled_count": p_remain,
        "persistent_swa_remains_not_entitled_rate": ratio(p_remain, len(persistent)),
        "persistent_mean_logit_remains_not_entitled_count": count(lambda r: r["mean_logit_prediction"] == "NOT_ENTITLED", persistent),
        "persistent_mean_logit_remains_not_entitled_rate": ratio(count(lambda r: r["mean_logit_prediction"] == "NOT_ENTITLED", persistent), len(persistent))}
    return result, outlier_row, support_row


PAIR_METRICS = (
    "epoch20_clean_ce", "epoch20_accuracy", "epoch20_macro_f1", "epoch20_support_recall",
    "epoch20_false_entitlement_total", "epoch20_false_not_entitled_total",
    "epoch20_polarity_error_total", "swa_clean_ce", "swa_accuracy", "swa_macro_f1",
    "swa_support_recall", "swa_false_entitlement_total", "swa_false_not_entitled_total",
    "swa_polarity_error_total", "clean_ce_delta", "accuracy_delta", "macro_f1_delta",
    "support_recall_delta", "polarity_error_delta", "total_corrected_rows",
    "total_harmed_rows", "net_correctness_change",
    "temporal_outlier_count", "swa_rescue_count", "swa_harm_count",
    "temporal_outlier_net_correctness_change", "target_swa_support_rescue_rate",
    "persistent_swa_support_rescue_rate", "false_entitlement_delta",
    "false_not_entitled_delta",
)


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def aggregate(run_summaries: list[dict[str, Any]], support_rows: list[dict[str, Any]]) -> tuple[
        dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    support_by_run = {row["run"]: row for row in support_rows}
    combined = {row["run"]: {**row, **support_by_run[row["run"]]} for row in run_summaries}
    pair_rows: list[dict[str, Any]] = []
    pair_dicts: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = combined[f"seed{seed}_baseline"]
        intervention = combined[f"seed{seed}_intervention"]
        seed_record: dict[str, Any] = {"seed": seed}
        for metric in PAIR_METRICS:
            left, right = baseline[metric], intervention[metric]
            delta = None if left is None or right is None else right - left
            pair_rows.append({"seed": seed, "metric": metric, "baseline": left,
                              "intervention": right, "intervention_minus_baseline": delta})
            seed_record[metric] = {"baseline": left, "intervention": right,
                                   "intervention_minus_baseline": delta}
        pair_dicts.append(seed_record)
    arms: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in run_summaries if row["arm"] == arm]
        summaries: dict[str, Any] = {}
        numeric_keys = [key for key in RUN_HEADER if key not in ("run", "seed", "arm",
                        "epoch20_pred_counts", "swa_pred_counts")]
        for key in numeric_keys:
            values = [row[key] for row in rows if row[key] is not None]
            if values:
                summaries[key] = {"mean": mean64(values), "median": statistics.median(values),
                    "minimum": min(values), "maximum": max(values),
                    "positive_seed_count": sum(float(value) > 0 for value in values),
                    "zero_seed_count": sum(float(value) == 0 for value in values),
                    "negative_seed_count": sum(float(value) < 0 for value in values)}
        arms[arm] = {"run_count": 3, "descriptive_statistics": summaries,
            "pooled": {"row_count": sum(row["row_count"] for row in rows),
                "temporal_outlier_count": sum(row["temporal_outlier_count"] for row in rows),
                "swa_rescue_count": sum(row["swa_rescue_count"] for row in rows),
                "swa_harm_count": sum(row["swa_harm_count"] for row in rows),
                "swa_unchanged_count": sum(row["swa_unchanged_count"] for row in rows),
                "temporal_outlier_net_correctness_change": sum(
                    row["temporal_outlier_net_correctness_change"] for row in rows),
                "false_entitlement_delta": sum(row["false_entitlement_delta"] for row in rows),
                "false_not_entitled_delta": sum(row["false_not_entitled_delta"] for row in rows),
                "polarity_error_delta": sum(row["polarity_error_delta"] for row in rows),
                "target_support_consensus_outlier_count": sum(
                    support_by_run[row["run"]]["target_support_consensus_outlier_count"] for row in rows),
                "persistent_stable_support_negative_count": sum(
                    support_by_run[row["run"]]["persistent_stable_support_negative_count"] for row in rows)},
            "seed_directions": {key: [sign(float(row[key])) for row in rows]
                for key in ("temporal_outlier_net_correctness_change", "false_entitlement_delta",
                            "false_not_entitled_delta", "polarity_error_delta", "macro_f1_delta")}}
    overall = {key: arms["baseline"]["pooled"][key] + arms["intervention"]["pooled"][key]
               for key in arms["baseline"]["pooled"]}
    overall["target_support_consensus_outlier_count"] = sum(
        row["target_support_consensus_outlier_count"] for row in support_rows)
    return arms, pair_rows, {"paired_seed_records": pair_dicts, **overall}


def decide(arms: dict[str, Any], overall: dict[str, Any],
           gates: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    pooled = {arm: arms[arm]["pooled"] for arm in ARMS}
    directions = {arm: arms[arm]["seed_directions"] for arm in ARMS}
    harm = all(pooled[arm]["temporal_outlier_net_correctness_change"] < 0
        and directions[arm]["temporal_outlier_net_correctness_change"].count(-1) >= 2
        and pooled[arm]["swa_harm_count"] > pooled[arm]["swa_rescue_count"] for arm in ARMS)
    primary_support = all(pooled[arm]["temporal_outlier_net_correctness_change"] > 0
        and directions[arm]["temporal_outlier_net_correctness_change"].count(1) >= 2
        and pooled[arm]["swa_harm_count"] < pooled[arm]["swa_rescue_count"] for arm in ARMS)
    boundary_safe = all(pooled[arm]["false_entitlement_delta"] <= 0
                        and pooled[arm]["polarity_error_delta"] <= 0 for arm in ARMS)
    inconsistent = any(len(set(directions[arm][metric])) > 1 for arm in ARMS for metric in
                       ("false_entitlement_delta", "false_not_entitled_delta",
                        "polarity_error_delta"))
    tradeoff = any(pooled[arm]["false_entitlement_delta"] > 0
        or pooled[arm]["false_not_entitled_delta"] > 0
        or arms[arm]["descriptive_statistics"]["macro_f1_delta"]["mean"] < 0
        for arm in ARMS) or inconsistent or not boundary_safe
    no_support = (not harm and not primary_support and (
        overall["temporal_outlier_net_correctness_change"] <= 0
        or (overall["swa_rescue_count"] == 0 and overall["swa_unchanged_count"] * 2 >=
            overall["temporal_outlier_count"])))
    selected = (REPLICATED_HARM if harm else BOUNDARY_TRADEOFF if primary_support and tradeoff
                else REPLICATED_SUPPORT if primary_support else NO_SUPPORT if no_support else MIXED)
    observed = {"arm_pooled": pooled, "arm_seed_directions": directions,
        "overall_pooled_net_correctness_change": overall["temporal_outlier_net_correctness_change"],
        "overall_target_support_consensus_outlier_count": overall[
            "target_support_consensus_outlier_count"], "replicated_harm_condition": harm,
        "primary_support_condition": primary_support, "boundary_safe": boundary_safe,
        "boundary_tradeoff_trigger": tradeoff, "no_support_condition": no_support}
    conditions = {BLOCKED: False, REPLICATED_HARM: harm,
                  REPLICATED_SUPPORT: primary_support and not tradeoff,
                  BOUNDARY_TRADEOFF: primary_support and tradeoff,
                  NO_SUPPORT: no_support, MIXED: not any((harm, primary_support, no_support))}
    texts = {BLOCKED: "integrity or calculation failure",
        REPLICATED_HARM: "both arms negative with >=2/3 negative seeds and pooled harms > rescues",
        REPLICATED_SUPPORT: "replicated positive temporal endpoint with boundary safety and no trade-off",
        BOUNDARY_TRADEOFF: "replicated positive temporal endpoint with one or more boundary trade-off triggers",
        NO_SUPPORT: "neither replicated support nor harm and nonpositive/unchanged causal response",
        MIXED: "all other integrity-passing arm, seed, temporal, or boundary mixtures"}
    for decision in DECISIONS:
        required = decision == selected
        condition = conditions[decision]
        gates.append({"decision": decision, "taxonomy_condition": texts[decision],
                      "required": required, "observed": {**observed, "condition": condition},
                      "passed": required == condition})
    if sum(row["required"] is True for row in gates) != 1 or any(row["passed"] is not True for row in gates):
        raise RuntimeError("decision gate is not exclusive and exhaustive")
    return selected, observed


REPORT_KEYS = {"stage", "decision", "runnable", "blocking_reasons", "diagnostic_only",
    "artifact_only_analysis", "training_performed", "model_loaded", "tokenizer_loaded",
    "checkpoint_loaded", "state_capsule_loaded", "external_data_used",
    "stage195c_runtime_repository_commit", "stage195a_runtime_repository_commit",
    "trainer_blob_commit", "trainer_blob_sha256", "canonical_labels", "source_epochs",
    "ordered_runs", "stage195a_directory", "stage195b_run_root", "row_transition_count",
    "all_twenty_exports_validated_per_run", "run_summaries", "arm_aggregates",
    "paired_seed_deltas", "overall_pooled", "decision_taxonomy", "production_swa_selected",
    "model_advancement_authorized", "subsequent_training_authorized",
    "entitlement_correction_implemented", "calibration_authorized", "ema_authorized",
    "statistical_significance_claimed", "external_generalization_claimed",
    "parameter_averaging_adopted_as_final_architecture", "interpretation_restrictions", "exception"}


def restrictions() -> list[str]:
    return ["clean controlled-dev causal diagnostic only", "artifact-only analysis",
        "no model/tokenizer/checkpoint/state-capsule loading", "no training",
        "no production SWA selection", "no model advancement", "no subsequent training",
        "no entitlement correction", "no calibration", "no EMA",
        "no statistical-significance claim", "no external-generalization claim",
        "parameter averaging not adopted as final architecture"]


def analyze(args: argparse.Namespace, tables: dict[str, list[dict[str, Any]]]) -> tuple[
        dict[str, Any], list[dict[str, Any]]]:
    repo, stage195a, run_root, _ = establish_safe_paths(args)
    validate_code_identity(repo, args.current_diagnostic_git_commit, tables["closure"])
    _, manifests = validate_stage195a(stage195a, run_root, tables["closure"])
    data = validate_runs(manifests, run_root, tables["closure"])
    transition_rows = decompose(data)
    summaries: list[dict[str, Any]] = []
    for run in RUNS:
        run_rows = [row for row in transition_rows if row["run"] == run]
        summary, outlier, support = summary_for_run(run, run_rows)
        summaries.append(summary)
        tables["runs"].append(summary)
        tables["outliers"].append(outlier)
        tables["support"].append(support)
    arms, pair_rows, overall = aggregate(summaries, tables["support"])
    tables["pairs"].extend(pair_rows)
    decision, taxonomy = decide(arms, overall, tables["decision"])
    report = {"stage": "Stage195-C", "decision": decision, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True, "artifact_only_analysis": True,
        "training_performed": False, "model_loaded": False, "tokenizer_loaded": False,
        "checkpoint_loaded": False, "state_capsule_loaded": False, "external_data_used": False,
        "stage195c_runtime_repository_commit": args.current_diagnostic_git_commit,
        "stage195a_runtime_repository_commit": STAGE195A_RUNTIME_COMMIT,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_BLOB_SHA256,
        "canonical_labels": list(LABELS), "source_epochs": list(SOURCE_EPOCHS),
        "ordered_runs": list(RUNS), "stage195a_directory": str(stage195a),
        "stage195b_run_root": str(run_root), "row_transition_count": len(transition_rows),
        "all_twenty_exports_validated_per_run": True, "run_summaries": summaries,
        "arm_aggregates": arms, "paired_seed_deltas": overall.pop("paired_seed_records"),
        "overall_pooled": overall, "decision_taxonomy": taxonomy,
        "production_swa_selected": False, "model_advancement_authorized": False,
        "subsequent_training_authorized": False, "entitlement_correction_implemented": False,
        "calibration_authorized": False, "ema_authorized": False,
        "statistical_significance_claimed": False, "external_generalization_claimed": False,
        "parameter_averaging_adopted_as_final_architecture": False,
        "interpretation_restrictions": restrictions(), "exception": None}
    exact_keys(report, REPORT_KEYS, "READY report")
    return report, transition_rows


def blocked_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    report = {"stage": "Stage195-C", "decision": BLOCKED, "runnable": False,
        "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "diagnostic_only": True,
        "artifact_only_analysis": True, "training_performed": False, "model_loaded": False,
        "tokenizer_loaded": False, "checkpoint_loaded": False, "state_capsule_loaded": False,
        "external_data_used": False,
        "stage195c_runtime_repository_commit": args.current_diagnostic_git_commit,
        "stage195a_runtime_repository_commit": STAGE195A_RUNTIME_COMMIT,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_BLOB_SHA256,
        "canonical_labels": list(LABELS), "source_epochs": list(SOURCE_EPOCHS),
        "ordered_runs": list(RUNS), "stage195a_directory": str(args.stage195a_dir.resolve()),
        "stage195b_run_root": str(args.stage195b_run_root.resolve()), "row_transition_count": 0,
        "all_twenty_exports_validated_per_run": False, "run_summaries": [],
        "arm_aggregates": {}, "paired_seed_deltas": [], "overall_pooled": {},
        "decision_taxonomy": {"selected_decision": BLOCKED},
        "production_swa_selected": False, "model_advancement_authorized": False,
        "subsequent_training_authorized": False, "entitlement_correction_implemented": False,
        "calibration_authorized": False, "ema_authorized": False,
        "statistical_significance_claimed": False, "external_generalization_claimed": False,
        "parameter_averaging_adopted_as_final_architecture": False,
        "interpretation_restrictions": restrictions(),
        "exception": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()}}
    exact_keys(report, REPORT_KEYS, "BLOCKED report")
    return report


def csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) \
        if isinstance(value, (dict, list, tuple)) else value


def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError(f"generated CSV exact schema mismatch: {set(row) ^ set(header)}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return stream.getvalue()


def markdown(report: dict[str, Any]) -> str:
    return "\n".join(["# Stage195-C parameter-SWA causal analysis", "",
        f"Decision: `{report['decision']}`", "", f"- Runnable: {str(report['runnable']).lower()}",
        "- Diagnostic only: true", "- Artifact-only: true", "- Training performed: false",
        "- Model/tokenizer/checkpoint/state capsule loaded: false",
        f"- Stage195-C runtime repository commit: `{report['stage195c_runtime_repository_commit']}`",
        f"- Stage195-A runtime repository commit: `{report['stage195a_runtime_repository_commit']}`",
        f"- Trainer blob commit: `{report['trainer_blob_commit']}`",
        f"- Trainer blob SHA256: `{report['trainer_blob_sha256']}`", "",
        "The result is limited to late-epoch parameter averaging on the frozen controlled dev artifacts.",
        "It authorizes no production selection, model advancement, calibration, entitlement correction, or training.", ""])


def render_outputs(report: dict[str, Any], transition_rows: list[dict[str, Any]],
                   tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    exact_keys(report, REPORT_KEYS, "rendered report")
    ready = report["decision"] != BLOCKED
    if ready and len(transition_rows) != 4320:
        raise ValueError("READY row-transition count mismatch")
    if not ready and transition_rows:
        raise ValueError("BLOCKED row-transition JSONL must be empty")
    row_text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                       for row in transition_rows)
    contents = {OUTPUTS["json"]: json.dumps(report, indent=2, sort_keys=True,
        ensure_ascii=False) + "\n", OUTPUTS["md"]: markdown(report), OUTPUTS["rows"]: row_text}
    for name, header in CSV_HEADERS.items():
        contents[OUTPUTS[name]] = render_csv(header, tables[name])
    if set(contents) != set(OUTPUTS.values()):
        raise RuntimeError("exact nine-output name closure mismatch")
    return contents


def publish(output: Path, contents: dict[str, str]) -> None:
    temporary = {name: output / f".{name}.stage195c.tmp" for name in contents}
    targets = {name: output / name for name in contents}
    if any(path.exists() for path in [*temporary.values(), *targets.values()]):
        raise FileExistsError("Stage195-C refuses to overwrite an output")
    try:
        for name, value in contents.items():
            with temporary[name].open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        report_name = OUTPUTS["json"]
        for name in [name for name in contents if name != report_name] + [report_name]:
            os.replace(temporary[name], targets[name])
    except BaseException:
        for path in [*temporary.values(), *targets.values()]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main() -> int:
    args = parse_args()
    try:
        *_, output = establish_safe_paths(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 2
    output.mkdir(parents=False, exist_ok=True)
    tables = {name: [] for name in CSV_HEADERS}
    try:
        report, transitions = analyze(args, tables)
        contents = render_outputs(report, transitions, tables)
        publish(output, contents)
        return 0
    except BaseException as exc:
        tables = {name: [] for name in CSV_HEADERS}
        tables["closure"].append({"scope": "failure", "run": "", "gate": "fail_closed_exception",
            "required": "no exception", "observed": {"type": type(exc).__name__, "message": str(exc)},
            "passed": False, "blocking_reason": f"{type(exc).__name__}: {exc}"})
        tables["decision"].append({"decision": BLOCKED,
            "taxonomy_condition": "provenance, schema, hash, cardinality, alignment, or calculation failure",
            "required": True, "observed": {"type": type(exc).__name__, "message": str(exc)},
            "passed": True})
        report = blocked_report(args, exc)
        try:
            publish(output, render_outputs(report, [], tables))
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            return 3
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
