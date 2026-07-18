#!/usr/bin/env python3
"""Localize Stage193 entitlement-boundary behavior from frozen artifacts only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE193_RUNTIME_COMMIT = "89a9805d0e9c877774f9ce4b356297d31645b74b"
TRAINER_BLOB_COMMIT = "e83d8af756fa84b7a91c14e0910ae388b07b5f02"
TRAINER_SHA256 = "25d42bdcd204219a2b2e5e7bf2a8b14459eafb4945c05c61ab3611bc9e7365bc"
STAGE193A_READY = "STAGE193A_TAIL3_FRESH_SEED_MANIFEST_READY"
STAGE193C_REPLICATED = "STAGE193C_TAIL3_SMOOTHING_REPLICATED"
STAGE193C_BLOCKED = "STAGE193C_TAIL3_FRESH_SEED_REPLICATION_BLOCKED"
STAGE193C_PARTIAL = "STAGE193C_TAIL3_SMOOTHING_PARTIAL_SIGNAL"
STAGE193C_NOT_REPLICATED = "STAGE193C_TAIL3_SMOOTHING_NOT_REPLICATED"

BLOCKED = "STAGE194A_ENTITLEMENT_LOCALIZATION_BLOCKED"
TEMPORAL_DOMINANT = "STAGE194A_TEMPORAL_CONSENSUS_OUTLIER_DOMINANT"
MAGNITUDE_DOMINANT = "STAGE194A_MEAN_MAGNITUDE_OUTLIER_DOMINANT"
PERSISTENT_DOMINANT = "STAGE194A_PERSISTENT_ENTITLEMENT_BIAS_DOMINANT"
MIXED = "STAGE194A_MIXED_TEMPORAL_AND_BOUNDARY_MECHANISMS"
INCONCLUSIVE = "STAGE194A_ENTITLEMENT_MECHANISM_INCONCLUSIVE"
DECISIONS = (BLOCKED, TEMPORAL_DOMINANT, MAGNITUDE_DOMINANT,
             PERSISTENT_DOMINANT, MIXED, INCONCLUSIVE)

LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
SEEDS = (177, 178, 179)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
EPOCHS = tuple(range(1, 21))
LATE_EPOCHS = (18, 19, 20)
SOURCES = ("selected", "epoch18", "epoch19", "epoch20", "tail3_mean", "tail3_median")

STAGE193A_OUTPUTS = {
    "stage193a_tail3_fresh_seed_manifest_report.json",
    "stage193a_tail3_fresh_seed_manifest_report.md",
    "stage193a_run_manifest.jsonl",
    "stage193a_run_command_matrix.csv",
    "stage193a_source_and_template_gate.csv",
    "stage193a_precommitted_gate.csv",
}
STAGE193C_OUTPUTS = {
    "stage193c_tail3_fresh_seed_replication_report.json",
    "stage193c_tail3_fresh_seed_replication_report.md",
    "stage193c_stage192a_closure_gate.csv",
    "stage193c_run_identity_gate.csv",
    "stage193c_epoch_metric_reconstruction.csv",
    "stage193c_comparator_metrics_by_seed.csv",
    "stage193c_comparator_aggregate_fresh.csv",
    "stage193c_comparator_aggregate_pooled.csv",
    "stage193c_pair_transition_summary.csv",
    "stage193c_pair_transition_by_gold.csv",
    "stage193c_primary_criterion_gate.csv",
    "stage193c_precommitted_decision_gate.csv",
}
OUTPUTS = {
    "json": "stage194a_entitlement_boundary_localization_report.json",
    "md": "stage194a_entitlement_boundary_localization_report.md",
    "closure": "stage194a_stage193_closure_gate.csv",
    "identity": "stage194a_run_identity_gate.csv",
    "rows": "stage194a_row_margin_decomposition.csv",
    "support": "stage194a_support_recall_decomposition.csv",
    "mechanisms": "stage194a_boundary_mechanism_summary.csv",
    "gold": "stage194a_gold_conditioned_margin_summary.csv",
    "patterns": "stage194a_temporal_pattern_summary.csv",
    "counterfactual": "stage194a_diagnostic_counterfactual_summary.csv",
    "criteria": "stage194a_mechanism_criterion_gate.csv",
    "decision": "stage194a_precommitted_decision_gate.csv",
}

CSV_HEADERS = {
    "closure": ["gate", "required", "observed", "passed", "blocking_reason"],
    "identity": ["run", "gate", "required", "observed", "passed", "blocking_reason"],
    "rows": ["run", "training_seed", "split_seed", "arm", "dev_position", "gold_label",
        "selected_epoch", "selected_logits", "epoch18_logits", "epoch19_logits", "epoch20_logits",
        "tail3_mean_logits", "tail3_median_logits", "selected_prediction", "epoch18_prediction",
        "epoch19_prediction", "epoch20_prediction", "tail3_mean_prediction", "tail3_median_prediction",
        "selected_entitlement_margin", "epoch18_entitlement_margin", "epoch19_entitlement_margin",
        "epoch20_entitlement_margin", "tail3_mean_entitlement_margin",
        "tail3_median_entitlement_margin", "selected_refute_margin", "epoch18_refute_margin",
        "epoch19_refute_margin", "epoch20_refute_margin", "tail3_mean_refute_margin",
        "tail3_median_refute_margin", "late_entitlement_margin_population_stddev",
        "late_entitlement_margin_range", "consecutive_sign_change_count", "support_sign_vote_count",
        "not_entitled_sign_vote_count", "exact_zero_tie_count", "sign_pattern",
        "selected_to_mean_transition", "selected_to_median_transition", "refute_involved",
        "selected_consensus_outlier", "mean_magnitude_override", "median_rescue",
        "persistent_stable_negative", "temporally_mixed_negative"],
    "support": ["scope", "aggregate", "run_count", "gold_support_count",
        "selected_support_true_positives", "tail3_mean_support_true_positives",
        "tail3_median_support_true_positives", "selected_support_recall", "tail3_mean_support_recall",
        "tail3_median_support_recall", "selected_to_mean_support_losses",
        "selected_to_mean_support_gains", "net_support_true_positive_change",
        "losses_to_NOT_ENTITLED", "losses_to_REFUTE", "gains_from_NOT_ENTITLED",
        "gains_from_REFUTE", "selected_consensus_outlier_count", "mean_magnitude_override_count",
        "median_rescue_count", "persistent_stable_negative_count", "temporally_mixed_negative_count",
        "tail3_mean_NOT_ENTITLED_false_negative_count", "tail3_mean_REFUTE_false_negative_count",
        "refute_involved_gold_support_count"],
    "mechanisms": ["scope", "aggregate", "run_count",
        "selected_to_mean_support_losses_to_NOT_ENTITLED", "selected_consensus_outlier_count",
        "mean_magnitude_override_count", "median_rescue_count",
        "tail3_mean_NOT_ENTITLED_false_negative_count", "persistent_stable_negative_count",
        "temporally_mixed_negative_count", "consensus_outlier_denominator", "consensus_outlier_share",
        "magnitude_override_denominator", "magnitude_override_share", "median_rescue_denominator",
        "median_rescue_rate", "persistent_bias_denominator", "persistent_bias_share",
        "refute_involved_support_loss_count"],
    "gold": ["scope", "aggregate", "run_count", "gold_label", "source", "count",
        "mean_entitlement_margin", "median_entitlement_margin", "population_stddev", "minimum",
        "maximum", "negative_margin_fraction", "positive_margin_fraction", "exact_zero_fraction",
        "pred_REFUTE", "pred_NOT_ENTITLED", "pred_SUPPORT"],
    "patterns": ["scope", "aggregate", "run_count", "gold_label", "sign_pattern", "count",
        "fraction", "mean_margin_range", "mean_margin_population_stddev",
        "selected_to_mean_transition_counts", "selected_to_median_transition_counts"],
    "counterfactual": ["scope", "aggregate", "run_count", "gold_support_count",
        "tail3_mean_support_true_positives", "tail3_median_support_true_positives",
        "tail3_mean_support_recall", "tail3_median_support_recall", "mean_to_median_support_gains",
        "mean_to_median_support_losses", "mean_NOT_ENTITLED_to_median_SUPPORT",
        "mean_REFUTE_to_median_SUPPORT", "mean_SUPPORT_to_median_NOT_ENTITLED",
        "mean_SUPPORT_to_median_REFUTE", "mean_magnitude_override_count", "median_rescue_count",
        "median_rescue_rate", "diagnostic_only"],
    "criteria": ["mechanism", "evidence_level", "criterion", "required", "observed", "passed"],
    "decision": ["decision", "taxonomy_condition", "required", "observed", "passed"],
}

STAGE194_SOURCE_FILES = (
    "reports/stage194a_entitlement_boundary_localization_spec.md",
    "scripts/analyze_stage194a_entitlement_boundary_localization.py",
)
STAGE193_FROZEN_FILES = (
    "reports/stage193a_tail3_fresh_seed_replication_spec.md",
    "scripts/build_stage193a_tail3_fresh_seed_manifest.py",
    "scripts/analyze_stage193c_tail3_fresh_seed_replication.py",
    "scripts/train_controlled_v6b_minimal.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage193a-dir", type=Path, required=True)
    parser.add_argument("--stage193b-run-root", type=Path, required=True)
    parser.add_argument("--stage193c-dir", type=Path, required=True)
    parser.add_argument("--current-diagnostic-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL row")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def bool_csv(value: str) -> bool | None:
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    return None


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in header})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def exact_int(value: Any) -> bool:
    return type(value) is int


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_call(repo: Path, arguments: list[str], *, binary: bool = False,
             dirty: bool = False) -> Any:
    result = subprocess.run(["git", *arguments], cwd=repo, check=False,
                            capture_output=True, shell=False)
    if dirty:
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", errors="strict").strip()


def gate(rows: list[dict[str, Any]], name: str, required: Any, observed: Any,
         passed: bool, reason: str) -> None:
    rows.append({"gate": name, "required": required, "observed": observed,
                 "passed": passed, "blocking_reason": "" if passed else reason})
    if not passed:
        raise ValueError(reason)


def identity_gate(rows: list[dict[str, Any]], run: str, name: str, required: Any,
                  observed: Any, passed: bool, reason: str) -> None:
    rows.append({"run": run, "gate": name, "required": required, "observed": observed,
                 "passed": passed, "blocking_reason": "" if passed else reason})
    if not passed:
        raise ValueError(f"{run}: {reason}")


def establish_safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise ValueError("repo root is not a directory")
    reports = (repo / "reports").resolve()
    if not reports.is_dir():
        raise ValueError("reports directory is absent")
    stage193a = args.stage193a_dir.resolve()
    run_root = args.stage193b_run_root.resolve()
    stage193c = args.stage193c_dir.resolve()
    output = args.output_dir.resolve()
    supplied_paths = (repo, stage193a, run_root, stage193c, output)
    if len(set(supplied_paths)) != len(supplied_paths):
        raise ValueError("all supplied input and output paths must be distinct")
    if (stage193a.parent != reports or
            not stage193a.name.startswith("stage193a_tail3_fresh_seed_manifest_") or
            not stage193a.is_dir()):
        raise ValueError("Stage193-A directory is unsafe or absent")
    if (run_root.parent != reports or
            not run_root.name.startswith("stage193b_tail3_fresh_seed_runs_") or
            not run_root.is_dir()):
        raise ValueError("Stage193-B run root is unsafe or absent")
    if (stage193c.parent != reports or
            not stage193c.name.startswith("stage193c_tail3_fresh_seed_replication_") or
            not stage193c.is_dir()):
        raise ValueError("Stage193-C directory is unsafe or absent")
    if output.parent != reports or not output.name.startswith(
            "stage194a_entitlement_boundary_localization_"):
        raise ValueError("Stage194-A output path is unsafe")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Stage194-A output exists and is nonempty")
    return repo, stage193a, run_root, stage193c, output


def validate_source_identity(repo: Path, supplied_commit: str,
                             closure: list[dict[str, Any]]) -> dict[str, Any]:
    valid_commit = re.fullmatch(r"[0-9a-f]{40}", supplied_commit or "") is not None
    gate(closure, "stage194_commit_format", "lowercase hexadecimal length 40",
         supplied_commit, valid_commit, "Stage194 diagnostic commit format is invalid")
    head = git_call(repo, ["rev-parse", "HEAD"])
    gate(closure, "stage194_commit_equals_head", supplied_commit, head,
         head == supplied_commit, "Stage194 diagnostic commit differs from HEAD")
    stage194_files: dict[str, Any] = {}
    for relative in STAGE194_SOURCE_FILES:
        current = (repo / relative).read_bytes()
        blob = git_call(repo, ["show", f"{supplied_commit}:{relative}"], binary=True)
        unstaged = git_call(repo, ["diff", "--quiet", "--", relative], dirty=True) == 0
        staged = git_call(repo, ["diff", "--cached", "--quiet", "--", relative], dirty=True) == 0
        passed = current == blob and unstaged and staged
        stage194_files[relative] = {
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "bytes_equal": current == blob, "unstaged_clean": unstaged, "staged_clean": staged,
        }
        gate(closure, f"stage194_source_identity:{relative}",
             {"bytes_equal": True, "unstaged_clean": True, "staged_clean": True},
             stage194_files[relative], passed, f"Stage194 source identity failed for {relative}")
    stage193_files: dict[str, Any] = {}
    for relative in STAGE193_FROZEN_FILES:
        current = (repo / relative).read_bytes()
        blob = git_call(repo, ["show", f"{STAGE193_RUNTIME_COMMIT}:{relative}"], binary=True)
        passed = current == blob
        stage193_files[relative] = {
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "runtime_blob_sha256": hashlib.sha256(blob).hexdigest(), "bytes_equal": passed,
        }
        gate(closure, f"stage193_runtime_blob_identity:{relative}", True,
             stage193_files[relative], passed, f"Stage193 frozen bytes differ for {relative}")
    trainer_relative = "scripts/train_controlled_v6b_minimal.py"
    trainer_current = (repo / trainer_relative).read_bytes()
    trainer_blob = git_call(repo, ["show", f"{TRAINER_BLOB_COMMIT}:{trainer_relative}"], binary=True)
    trainer_sha = hashlib.sha256(trainer_current).hexdigest()
    trainer_ok = (trainer_current == trainer_blob and trainer_sha == TRAINER_SHA256 and
                  stage193_files[trainer_relative]["bytes_equal"])
    trainer = {"path": str((repo / trainer_relative).resolve()),
        "blob_commit": TRAINER_BLOB_COMMIT, "current_sha256": trainer_sha,
        "commit_blob_sha256": hashlib.sha256(trainer_blob).hexdigest(),
        "bytes_equal": trainer_current == trainer_blob}
    gate(closure, "frozen_trainer_blob_and_sha256",
         {"blob_commit": TRAINER_BLOB_COMMIT, "sha256": TRAINER_SHA256}, trainer,
         trainer_ok, "trainer blob or SHA256 differs from the frozen identity")
    return {"supplied_commit": supplied_commit, "repository_head": head,
            "stage194_files": stage194_files, "stage193_runtime_files": stage193_files,
            "trainer": trainer, "passed": True}


def option_map(argv: Any) -> dict[str, Any]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a string list")
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            raise ValueError(f"unsupported argv token {token!r}")
        key = token[2:].replace("-", "_")
        if key in result:
            raise ValueError(f"duplicate argv option {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[key] = argv[index + 1]
            index += 2
        else:
            result[key] = True
            index += 1
    return result


def validate_stage193a(stage193a: Path, run_root: Path,
                       closure: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = {path.name for path in stage193a.iterdir()}
    exact_outputs = entries == STAGE193A_OUTPUTS and all(
        (stage193a / name).is_file() for name in STAGE193A_OUTPUTS)
    gate(closure, "stage193a_exact_six_outputs", sorted(STAGE193A_OUTPUTS), sorted(entries),
         exact_outputs, "Stage193-A exact six-output set mismatch")
    report = read_json(stage193a / "stage193a_tail3_fresh_seed_manifest_report.json")
    required = {"stage": "Stage193-A", "decision": STAGE193A_READY, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True,
        "exact_six_run_diagnostic_execution_authorized": True,
        "training_for_model_advancement_authorized": False,
        "model_advancement_decision": False, "subsequent_training_authorized": False,
        "external_data_used": False, "checkpoint_loaded": False, "model_loaded": False,
        "capsule_loaded": False, "statistical_significance_claim": False,
        "current_diagnostic_git_commit": STAGE193_RUNTIME_COMMIT,
        "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_SHA256,
        "ordered_runs": list(RUNS), "run_manifest_count": 6,
        "expected_trajectory_rows_per_run": 20, "expected_prediction_exports_per_run": 20,
        "expected_prediction_rows_per_export": 720, "expected_state_capsules_per_run": 0,
        "canonical_labels": list(LABELS), "logits_source": 'output["logits"]'}
    report_ok = type(report) is dict and all(report.get(key) == value for key, value in required.items())
    integer_contract = {"run_manifest_count": 6, "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20, "expected_prediction_rows_per_export": 720,
        "expected_state_capsules_per_run": 0}
    report_ok = report_ok and all(exact_int(report.get(key)) and report.get(key) == value
                                  for key, value in integer_contract.items())
    gate(closure, "stage193a_ready_closure", required,
         {key: report.get(key) for key in required}, report_ok,
         "Stage193-A READY closure mismatch")
    source_identity = report.get("source_identity") or {}
    source_trainer = source_identity.get("trainer") or {}
    source_ok = (source_identity.get("passed") is True and
        source_identity.get("supplied_commit") == STAGE193_RUNTIME_COMMIT and
        source_identity.get("repository_head") == STAGE193_RUNTIME_COMMIT and
        source_trainer.get("blob_commit") == TRAINER_BLOB_COMMIT and
        source_trainer.get("current_sha256") == TRAINER_SHA256 and
        source_trainer.get("commit_blob_sha256") == TRAINER_SHA256 and
        source_trainer.get("bytes_equal") is True)
    gate(closure, "stage193a_source_identity", True, source_identity, source_ok,
         "Stage193-A source identity mismatch")
    stage193a_run_root = Path(str(report.get("stage193b_run_root", ""))).resolve()
    gate(closure, "stage193a_exact_stage193b_run_root", str(run_root), str(stage193a_run_root),
         stage193a_run_root == run_root, "Stage193-A records a different Stage193-B root")
    gate_header = ["gate", "required", "observed", "passed", "blocking_reason"]
    for filename in ("stage193a_source_and_template_gate.csv", "stage193a_precommitted_gate.csv"):
        header, rows = read_csv(stage193a / filename)
        passed = header == gate_header and bool(rows) and all(
            bool_csv(row.get("passed", "")) is True for row in rows)
        gate(closure, f"stage193a_all_passed:{filename}", True,
             {"header": header, "row_count": len(rows)}, passed,
             f"Stage193-A gate closure failed for {filename}")
    matrix_header, matrix_rows = read_csv(stage193a / "stage193a_run_command_matrix.csv")
    required_matrix_header = ["run", "training_seed", "split_seed", "arm",
        "planned_run_directory", "planned_output_json_path", "planned_selected_checkpoint_path",
        "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
        "runtime_repository_commit", "command", "expected_trajectory_rows",
        "expected_prediction_exports", "expected_prediction_rows_per_export",
        "expected_state_capsules"]
    matrix_ok = (matrix_header == required_matrix_header and len(matrix_rows) == 6 and
                 [row.get("run") for row in matrix_rows] == list(RUNS))
    gate(closure, "stage193a_command_matrix_schema_order", required_matrix_header,
         {"header": matrix_header, "runs": [row.get("run") for row in matrix_rows]}, matrix_ok,
         "Stage193-A command matrix schema or order mismatch")
    manifests = read_jsonl(stage193a / "stage193a_run_manifest.jsonl")
    manifest_order_ok = len(manifests) == 6 and [row.get("run") for row in manifests] == list(RUNS)
    gate(closure, "stage193a_manifest_six_run_order", list(RUNS),
         [row.get("run") for row in manifests], manifest_order_ok,
         "Stage193-A manifest order or cardinality mismatch")
    frozen = report.get("frozen_source_identities") or {}
    for manifest, matrix_row, run in zip(manifests, matrix_rows, RUNS):
        seed = int(run[4:7])
        arm = run.split("_", 1)[1]
        run_dir = (run_root / run).resolve()
        required_row = {"stage": "Stage193-A", "run": run, "training_seed": seed,
            "split_seed": 174, "arm": arm, "canonical_labels": list(LABELS),
            "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_SHA256,
            "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
            "planned_run_directory": str(run_dir),
            "planned_output_json_path": str((run_dir / "training_report.json").resolve()),
            "planned_selected_checkpoint_path": str((run_dir / "selected_checkpoint.pt").resolve()),
            "expected_trajectory_contract_path": str((run_dir / "stage191_trajectory_contract.json").resolve()),
            "expected_trajectory_ledger_path": str((run_dir / "stage191_trajectory_epoch_metrics.jsonl").resolve()),
            "expected_prediction_export_paths": [str((run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve()) for epoch in EPOCHS],
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0,
            "logits_source": 'output["logits"]', "runnable": True, "diagnostic_only": True,
            "exact_six_run_diagnostic_execution_authorized": True,
            "training_for_model_advancement_authorized": False,
            "model_advancement_decision": False, "subsequent_training_authorized": False,
            "external_data_used": False}
        if any(manifest.get(key) != value for key, value in required_row.items()):
            raise ValueError(f"{run}: Stage193-A manifest row mismatch")
        integer_fields = {"training_seed": seed, "split_seed": 174,
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0}
        if any(not exact_int(manifest.get(key)) or manifest.get(key) != value
               for key, value in integer_fields.items()):
            raise ValueError(f"{run}: Stage193-A manifest integer contract mismatch")
        matrix_required = {"training_seed": str(seed), "split_seed": "174",
            "arm": arm, "planned_run_directory": str(run_dir),
            "planned_output_json_path": str((run_dir / "training_report.json").resolve()),
            "planned_selected_checkpoint_path": str((run_dir / "selected_checkpoint.pt").resolve()),
            "trainer_source_path": source_trainer.get("path"),
            "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_SHA256,
            "runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
            "expected_trajectory_rows": "20", "expected_prediction_exports": "20",
            "expected_prediction_rows_per_export": "720", "expected_state_capsules": "0"}
        if any(matrix_row.get(key) != value for key, value in matrix_required.items()):
            raise ValueError(f"{run}: Stage193-A command matrix identity mismatch")
        options = option_map(manifest.get("argv"))
        argv_required = {"architecture": "v6b_minimal", "backbone": "mamba", "device": "cuda",
            "model_name": "state-spaces/mamba-130m-hf",
            "data": "data/controlled_v5_v3_without_time_swap.jsonl", "seed": str(seed),
            "split_seed": "174", "epochs": "20", "select_metric": "final_macro_f1",
            "selected_checkpoint_filename": "selected_checkpoint.pt",
            "stage193_tail3_fresh_seed_observability": True}
        if any(options.get(key) != value for key, value in argv_required.items()):
            raise ValueError(f"{run}: Stage193-A argv envelope mismatch")
        if ("stage191_trajectory_replay_observability" in options or
                "stage191_save_trajectory_state_capsules" in options):
            raise ValueError(f"{run}: Stage191 observability appears in Stage193 argv")
        output_required = {"output_json": run_dir / "training_report.json",
            "output_predictions_json": run_dir / "clean_dev_predictions.json",
            "stage115_clean_dev_scalar_output_jsonl": run_dir / "clean_dev_scalars.jsonl"}
        if any(Path(str(options.get(key, ""))).resolve() != path.resolve()
               for key, path in output_required.items()):
            raise ValueError(f"{run}: Stage193-A argv output path mismatch")
        sidecar = Path(str(frozen.get("stage185_sidecar_path", ""))).resolve()
        if arm == "baseline":
            arm_ok = (options.get("compatible_positive_margin_weight") in ("0", "0.0") and
                "controlled_integrity_sidecar_path" not in options and
                "expected_integrity_sidecar_semantic_sha256" not in options)
        else:
            arm_ok = (options.get("compatible_positive_margin_weight") == "0.05" and
                options.get("compatible_positive_margin_logit") == "0.0" and
                Path(str(options.get("controlled_integrity_sidecar_path", ""))).resolve() == sidecar and
                re.fullmatch(r"[0-9a-f]{64}", str(options.get(
                    "expected_integrity_sidecar_semantic_sha256", ""))) is not None and
                options.get("expected_integrity_sidecar_semantic_sha256") ==
                    frozen.get("stage185_sidecar_semantic_sha256"))
        if not arm_ok:
            raise ValueError(f"{run}: Stage193-A arm contract mismatch")
    return report, manifests


def rounded_equal(value: Any, expected: float) -> bool:
    return finite(value) and round(float(value), 6) == expected


def validate_stage193c(stage193c: Path, stage193a: Path, run_root: Path,
                       closure: list[dict[str, Any]]) -> dict[str, Any]:
    entries = {path.name for path in stage193c.iterdir()}
    exact_outputs = entries == STAGE193C_OUTPUTS and all(
        (stage193c / name).is_file() for name in STAGE193C_OUTPUTS)
    gate(closure, "stage193c_exact_twelve_outputs", sorted(STAGE193C_OUTPUTS), sorted(entries),
         exact_outputs, "Stage193-C exact twelve-output set mismatch")
    report = read_json(stage193c / "stage193c_tail3_fresh_seed_replication_report.json")
    required = {"stage": "Stage193-C", "decision": STAGE193C_REPLICATED, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True, "checkpoint_loaded": False,
        "model_loaded": False, "capsule_loaded": False, "external_data_used": False,
        "statistical_significance_claim": False, "model_advancement_decision": False,
        "subsequent_training_authorized": False,
        "current_diagnostic_git_commit": STAGE193_RUNTIME_COMMIT,
        "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_SHA256,
        "primary_candidate": "tail3_mean_logits", "tail2_descriptive_only": True}
    report_ok = type(report) is dict and all(report.get(key) == value for key, value in required.items())
    report_ok = report_ok and Path(str(report.get("stage193a_directory", ""))).resolve() == stage193a
    report_ok = report_ok and Path(str(report.get("stage193b_run_root", ""))).resolve() == run_root
    gate(closure, "stage193c_replicated_closure", required,
         {key: report.get(key) for key in required}, report_ok,
         "Stage193-C replicated closure mismatch")
    taxonomy = report.get("decision_taxonomy") or {}
    taxonomy_ok = (type(taxonomy) is dict and taxonomy.get("selected_decision") == STAGE193C_REPLICATED and
                   taxonomy.get("positive_conjunction") is True and
                   taxonomy.get("partial_conjunction") is False)
    gate(closure, "stage193c_report_decision_taxonomy", STAGE193C_REPLICATED, taxonomy,
         taxonomy_ok, "Stage193-C report taxonomy mismatch")
    criteria_header = ["criterion_group", "criterion", "required", "observed", "passed"]
    header, rows = read_csv(stage193c / "stage193c_primary_criterion_gate.csv")
    criteria_ok = header == criteria_header and bool(rows) and all(
        bool_csv(row.get("passed", "")) is True for row in rows)
    gate(closure, "stage193c_every_primary_criterion_passed", True,
         {"header": header, "row_count": len(rows)}, criteria_ok,
         "Stage193-C primary criterion gate failed")
    decision_header = ["decision", "taxonomy_condition", "required", "observed", "passed"]
    header, rows = read_csv(stage193c / "stage193c_precommitted_decision_gate.csv")
    expected_order = [STAGE193C_BLOCKED, STAGE193C_REPLICATED,
                      STAGE193C_PARTIAL, STAGE193C_NOT_REPLICATED]
    decision_ok = (header == decision_header and [row.get("decision") for row in rows] == expected_order and
        all(bool_csv(row.get("passed", "")) is True for row in rows) and
        sum(bool_csv(row.get("required", "")) is True for row in rows) == 1 and
        sum(bool_csv(row.get("observed", "")) is True for row in rows) == 1 and
        bool_csv(rows[1].get("required", "")) is True and
        bool_csv(rows[1].get("observed", "")) is True)
    gate(closure, "stage193c_only_replicated_decision_selected", STAGE193C_REPLICATED,
         {"header": header, "rows": rows}, decision_ok,
         "Stage193-C decision gate does not select only the replicated decision")
    fresh = report.get("fresh_aggregates") or {}
    independent = fresh.get("independent_selected") or {}
    tail3 = fresh.get("tail3_mean_logits") or {}
    expected_independent = {"mean_pair_clean_ce": 0.581720, "mean_pair_accuracy": 0.875694,
        "mean_pair_macro_f1": 0.828554, "mean_pair_support_recall": 0.679775,
        "support_delta_range": 33.0, "false_entitlement_delta_range": 29.0}
    expected_tail3 = {"mean_pair_clean_ce": 0.570286, "mean_pair_accuracy": 0.890509,
        "mean_pair_macro_f1": 0.820833, "mean_pair_support_recall": 0.516854,
        "support_delta_range": 6.0, "false_entitlement_delta_range": 1.0,
        "max_abs_refute_delta": 1.0, "max_abs_polarity_delta": 0.0}
    aggregate_ok = (all(rounded_equal(independent.get(key), value)
                        for key, value in expected_independent.items()) and
                    all(rounded_equal(tail3.get(key), value) for key, value in expected_tail3.items()))
    gate(closure, "stage193c_frozen_fresh_aggregates", {"independent_selected": expected_independent,
         "tail3_mean_logits": expected_tail3}, {"independent_selected": independent,
         "tail3_mean_logits": tail3}, aggregate_ok, "Stage193-C fresh aggregates differ")
    return report


def canonical_prediction(logits: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda index: logits[index])]


def exported_rows(path: Path, epoch: int, run: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 720:
        raise ValueError(f"{run} epoch {epoch}: expected 720 prediction rows")
    for position, row in enumerate(rows):
        if (not exact_int(row.get("epoch")) or row.get("epoch") != epoch or
                not exact_int(row.get("dev_position")) or row.get("dev_position") != position):
            raise ValueError(f"{run} epoch {epoch}: epoch/dev_position mismatch")
        if row.get("gold_final_label") not in LABELS or row.get("predicted_final_label") not in LABELS:
            raise ValueError(f"{run} epoch {epoch}: noncanonical label")
        logits = row.get("final_logits")
        if type(logits) is not list or len(logits) != 3 or any(not finite(value) for value in logits):
            raise ValueError(f"{run} epoch {epoch}: invalid final_logits")
        binary64_logits = [float(value) for value in logits]
        if row.get("predicted_final_label") != canonical_prediction(binary64_logits):
            raise ValueError(f"{run} epoch {epoch}: prediction differs from canonical argmax")
    return rows


def validate_runs(manifests: list[dict[str, Any]], run_root: Path,
                  identity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    root_entries = {path.name for path in run_root.iterdir()}
    if root_entries != set(RUNS) or any(not (run_root / run).is_dir() for run in RUNS):
        raise ValueError("Stage193-B root does not contain exactly the six frozen run directories")
    data: dict[str, Any] = {}
    global_golds: list[str] | None = None
    for manifest in manifests:
        run = manifest["run"]
        seed = manifest["training_seed"]
        arm = manifest["arm"]
        run_dir = (run_root / run).resolve()
        identity_gate(identity_rows, run, "exact_run_directory", str(run_dir), str(run_dir),
                      run_dir.is_dir(), "run directory absent")
        contract_path = Path(manifest["expected_trajectory_contract_path"]).resolve()
        ledger_path = Path(manifest["expected_trajectory_ledger_path"]).resolve()
        report_path = Path(manifest["planned_output_json_path"]).resolve()
        expected_required_paths = {
            "contract": (run_dir / "stage191_trajectory_contract.json").resolve(),
            "ledger": (run_dir / "stage191_trajectory_epoch_metrics.jsonl").resolve(),
            "training_report": (run_dir / "training_report.json").resolve(),
        }
        observed_required_paths = {"contract": contract_path, "ledger": ledger_path,
                                   "training_report": report_path}
        paths_ok = observed_required_paths == expected_required_paths and all(
            path.is_file() for path in observed_required_paths.values())
        identity_gate(identity_rows, run, "exact_required_artifact_paths",
                      {key: str(value) for key, value in expected_required_paths.items()},
                      {key: str(value) for key, value in observed_required_paths.items()}, paths_ok,
                      "required artifact path mismatch or absence")
        contract = read_json(contract_path)
        required_contract = {"observability_mode": "stage193_tail3_fresh_seed_replication",
            "authorized_training_seeds": list(SEEDS), "training_seed_authorized": True,
            "training_seed": seed, "split_seed": 174, "arm": arm, "epoch_count": 20,
            "expected_dev_rows": 720, "expected_state_capsules": 0,
            "canonical_logit_column_labels": list(LABELS), "logits_source": 'output["logits"]',
            "stage191_trajectory_observability_implementation_reused": True,
            "state_capsule_saving_enabled": False, "training_semantics_changed": False,
            "extra_forward_pass_performed": False, "loss_logits_used": False,
            "external_data_used": False, "trainer_source_commit": STAGE193_RUNTIME_COMMIT,
            "trainer_sha256": TRAINER_SHA256}
        contract_ints = {"training_seed": seed, "split_seed": 174, "epoch_count": 20,
                         "expected_dev_rows": 720, "expected_state_capsules": 0}
        authorized = contract.get("authorized_training_seeds")
        contract_ok = (type(contract) is dict and
            all(contract.get(key) == value for key, value in required_contract.items()) and
            type(authorized) is list and authorized == list(SEEDS) and
            all(exact_int(value) for value in authorized) and
            all(exact_int(contract.get(key)) and contract.get(key) == value
                for key, value in contract_ints.items()) and
            contract.get("enabled_flags") == {
                "stage191_trajectory_replay_observability": False,
                "stage191_save_trajectory_state_capsules": False,
                "stage193_tail3_fresh_seed_observability": True})
        identity_gate(identity_rows, run, "exact_stage193_observability_contract",
                      required_contract, contract, contract_ok, "Stage193 observability contract mismatch")
        training_report = read_json(report_path)
        if type(training_report) is not dict:
            raise ValueError(f"{run}: training report root is not an object")
        provenance_path = (run_dir / "run_provenance.json").resolve()
        provenance_path_ok = (Path(str(training_report.get("run_provenance_json", ""))).resolve() ==
                              provenance_path and provenance_path.is_file())
        identity_gate(identity_rows, run, "run_provenance_path", str(provenance_path),
                      training_report.get("run_provenance_json"), provenance_path_ok,
                      "run provenance path mismatch")
        report_runs = training_report.get("runs")
        if (type(report_runs) is not dict or set(report_runs) != {"single"} or
                type(report_runs.get("single")) is not dict):
            raise ValueError(f"{run}: training_report.runs.single schema mismatch")
        selected_epoch_report = training_report["runs"]["single"].get("best_epoch")
        final_epoch = training_report["runs"]["single"].get("final_epoch")
        if (not exact_int(selected_epoch_report) or selected_epoch_report not in EPOCHS or
                not exact_int(final_epoch) or final_epoch != 20):
            raise ValueError(f"{run}: selected/final epoch identity mismatch")
        provenance = read_json(provenance_path)
        raw_argv = provenance.get("raw_sys_argv")
        identity_gate(identity_rows, run, "exact_invoked_argv", manifest.get("argv"), raw_argv,
                      raw_argv == manifest.get("argv"), "runtime argv differs from Stage193-A")
        source = provenance.get("source_provenance") or {}
        source_ok = (type(source) is dict and source.get("git_commit") == STAGE193_RUNTIME_COMMIT and
                     source.get("trainer_sha256") == TRAINER_SHA256)
        identity_gate(identity_rows, run, "runtime_and_trainer_identity",
                      {"git_commit": STAGE193_RUNTIME_COMMIT, "trainer_sha256": TRAINER_SHA256},
                      source, source_ok, "runtime commit or trainer SHA256 mismatch")
        finalization = provenance.get("finalization")
        if type(finalization) is not dict:
            raise ValueError(f"{run}: finalization is not an object")
        selected_epoch = finalization.get("selected_epoch")
        completed_ok = ("completed_epochs" not in finalization or
                        (exact_int(finalization.get("completed_epochs")) and
                         finalization.get("completed_epochs") == 20))
        selection_ok = (exact_int(selected_epoch) and selected_epoch in EPOCHS and
                        selected_epoch == selected_epoch_report and completed_ok)
        identity_gate(identity_rows, run, "selected_and_final_epoch_identity",
                      {"selected_epoch": selected_epoch_report, "final_epoch": 20},
                      {"selected_epoch": selected_epoch,
                       "completed_epochs": finalization.get("completed_epochs"),
                       "final_epoch": final_epoch}, selection_ok,
                      "selected/final epoch alignment mismatch")
        split = provenance.get("split_seed_contract") or {}
        split_fields = {"training_seed": seed, "configured_split_seed": 174,
                        "resolved_split_seed": 174, "clean_main_dev_rows": 720}
        split_ok = (type(split) is dict and split.get("split_seed_explicit") is True and
                    all(exact_int(split.get(key)) and split.get(key) == value
                        for key, value in split_fields.items()) and contract.get("arm") == arm)
        identity_gate(identity_rows, run, "exact_run_seed_split_arm",
                      {"seed": seed, "split_seed": 174, "arm": arm},
                      {"seed": split.get("training_seed"), "split_seed": split.get("resolved_split_seed"),
                       "arm": contract.get("arm")}, split_ok, "run seed/split/arm mismatch")
        activity = ((provenance.get("data_provenance") or {}).get("auxiliary_activity") or {})
        inactive = ("stage57_active", "stage66_active", "stage75_active", "stage80a_active",
                    "time_swap_active", "external_evaluation_active")
        external_ok = (contract.get("external_data_used") is False and type(activity) is dict and
                       all(activity.get(key) is False for key in inactive))
        identity_gate(identity_rows, run, "no_external_or_auxiliary_data", True, activity,
                      external_ok, "external or auxiliary data activity present")
        capsule_names = [path.name for path in run_dir.iterdir() if path.is_file() and
                         re.fullmatch(r"stage191_trajectory_state_epoch_[0-9]+\.pt", path.name)]
        identity_gate(identity_rows, run, "zero_state_capsules", 0, len(capsule_names),
                      len(capsule_names) == 0, "state capsule files present")
        trajectory_rows = read_jsonl(ledger_path)
        trajectory_ok = (len(trajectory_rows) == 20 and
                         [row.get("epoch") for row in trajectory_rows] == list(EPOCHS) and
                         all(exact_int(row.get("epoch")) for row in trajectory_rows))
        identity_gate(identity_rows, run, "exact_twenty_epoch_ledger", list(EPOCHS),
                      [row.get("epoch") for row in trajectory_rows], trajectory_ok,
                      "trajectory ledger is not exact epochs 1 through 20")
        trajectory = {row["epoch"]: row for row in trajectory_rows}
        enumerated_exports = {path.name for path in run_dir.iterdir() if path.is_file() and
            re.fullmatch(r"stage191_dev_predictions_epoch_[0-9]{3}\.jsonl", path.name)}
        expected_names = {f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl" for epoch in EPOCHS}
        identity_gate(identity_rows, run, "exact_twenty_enumerated_prediction_exports",
                      sorted(expected_names), sorted(enumerated_exports),
                      enumerated_exports == expected_names,
                      "enumerated prediction export set mismatch")
        expected_paths = manifest.get("expected_prediction_export_paths")
        if (type(expected_paths) is not list or len(expected_paths) != 20 or
                any(type(value) is not str for value in expected_paths)):
            raise ValueError(f"{run}: Stage193-A expected export path list mismatch")
        retained: dict[int, list[dict[str, Any]]] = {}
        run_golds: list[str] | None = None
        for epoch, expected_path_text in zip(EPOCHS, expected_paths):
            prediction_path = Path(expected_path_text).resolve()
            ledger_row = trajectory[epoch]
            sha = file_sha256(prediction_path) if prediction_path.is_file() else None
            path_hash_ok = (prediction_path == (run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve() and
                Path(str(ledger_row.get("prediction_export_path", ""))).resolve() == prediction_path and
                re.fullmatch(r"[0-9a-f]{64}", str(ledger_row.get("prediction_export_sha256", ""))) is not None and
                sha == ledger_row.get("prediction_export_sha256"))
            if not path_hash_ok:
                raise ValueError(f"{run} epoch {epoch}: prediction path or SHA256 mismatch")
            ledger_int_ok = (exact_int(ledger_row.get("epoch")) and ledger_row.get("epoch") == epoch and
                exact_int(ledger_row.get("dev_row_count")) and ledger_row.get("dev_row_count") == 720 and
                exact_int(ledger_row.get("best_epoch_before")) and ledger_row.get("best_epoch_before") in range(0, epoch) and
                exact_int(ledger_row.get("best_epoch_after")) and ledger_row.get("best_epoch_after") in range(1, epoch + 1))
            if not ledger_int_ok:
                raise ValueError(f"{run} epoch {epoch}: ledger integer/cardinality mismatch")
            prediction_rows = exported_rows(prediction_path, epoch, run)
            golds = [row["gold_final_label"] for row in prediction_rows]
            if run_golds is None:
                run_golds = golds
            if golds != run_golds:
                raise ValueError(f"{run} epoch {epoch}: gold alignment mismatch")
            if epoch in LATE_EPOCHS or epoch == selected_epoch:
                retained[epoch] = prediction_rows
        if global_golds is None:
            global_golds = run_golds
        if run_golds != global_golds:
            raise ValueError(f"{run}: cross-run gold alignment mismatch")
        identity_gate(identity_rows, run, "all_exports_hash_cardinality_alignment", True, True,
                      True, "prediction export validation failed")
        data[run] = {"seed": seed, "arm": arm, "selected_epoch": selected_epoch,
                     "epochs": retained}
    return data


def mean64(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(float(value) for value in values) / len(values)


def median64(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


def population_stddev(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("population standard deviation requires values")
    mean = mean64(values)
    return math.sqrt(math.fsum((float(value) - mean) ** 2 for value in values) / len(values))


def entitlement_margin(logits: Sequence[float]) -> float:
    return float(logits[LABEL_INDEX["SUPPORT"]]) - float(logits[LABEL_INDEX["NOT_ENTITLED"]])


def refute_margin(logits: Sequence[float]) -> float:
    return float(logits[LABEL_INDEX["REFUTE"]]) - max(
        float(logits[LABEL_INDEX["NOT_ENTITLED"]]), float(logits[LABEL_INDEX["SUPPORT"]]))


def sign_symbol(value: float) -> str:
    if value > 0.0:
        return "+"
    if value < 0.0:
        return "-"
    return "0"


def decompose_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in RUNS:
        item = data[run]
        selected_epoch = item["selected_epoch"]
        epochs = item["epochs"]
        for position in range(720):
            originals = {epoch: epochs[epoch][position] for epoch in set((*LATE_EPOCHS, selected_epoch))}
            gold = originals[selected_epoch]["gold_final_label"]
            if any(originals[epoch]["gold_final_label"] != gold for epoch in originals):
                raise ValueError(f"{run} position {position}: retained gold alignment mismatch")
            selected_logits = [float(value) for value in originals[selected_epoch]["final_logits"]]
            late_logits = {epoch: [float(value) for value in originals[epoch]["final_logits"]]
                           for epoch in LATE_EPOCHS}
            mean_logits = [mean64([late_logits[epoch][index] for epoch in LATE_EPOCHS])
                           for index in range(3)]
            median_logits = [median64([late_logits[epoch][index] for epoch in LATE_EPOCHS])
                             for index in range(3)]
            logits_by_source = {"selected": selected_logits, "epoch18": late_logits[18],
                "epoch19": late_logits[19], "epoch20": late_logits[20],
                "tail3_mean": mean_logits, "tail3_median": median_logits}
            predictions = {source: canonical_prediction(logits)
                           for source, logits in logits_by_source.items()}
            margins = {source: entitlement_margin(logits)
                       for source, logits in logits_by_source.items()}
            refute_margins = {source: refute_margin(logits)
                              for source, logits in logits_by_source.items()}
            late_margins = [margins[f"epoch{epoch}"] for epoch in LATE_EPOCHS]
            signs = [sign_symbol(value) for value in late_margins]
            positive_votes = sum(value > 0.0 for value in late_margins)
            negative_votes = sum(value < 0.0 for value in late_margins)
            zero_votes = sum(value == 0.0 for value in late_margins)
            sign_changes = sum(signs[index] != signs[index - 1] for index in range(1, 3))
            refute_involved = any(prediction == "REFUTE" for prediction in predictions.values())
            relevant_no_zero = all(margins[source] != 0.0 for source in
                ("selected", "epoch18", "epoch19", "epoch20", "tail3_mean"))
            selected_consensus = False
            if selected_epoch in LATE_EPOCHS:
                other_epochs = [epoch for epoch in LATE_EPOCHS if epoch != selected_epoch]
                selected_consensus = (gold == "SUPPORT" and not refute_involved and
                    predictions["selected"] == "SUPPORT" and
                    predictions["tail3_mean"] == "NOT_ENTITLED" and
                    margins["selected"] > 0.0 and
                    all(margins[f"epoch{epoch}"] < 0.0 for epoch in other_epochs) and
                    relevant_no_zero)
            magnitude_override = (gold == "SUPPORT" and not refute_involved and
                predictions["selected"] == "SUPPORT" and
                predictions["tail3_mean"] == "NOT_ENTITLED" and positive_votes == 2 and
                negative_votes == 1 and margins["tail3_mean"] < 0.0 and relevant_no_zero)
            median_rescue = magnitude_override and predictions["tail3_median"] == "SUPPORT"
            persistent = (gold == "SUPPORT" and not refute_involved and
                predictions["tail3_mean"] == "NOT_ENTITLED" and negative_votes == 3 and
                all(predictions[f"epoch{epoch}"] == "NOT_ENTITLED" for epoch in LATE_EPOCHS) and
                all(margins[source] != 0.0 for source in
                    ("epoch18", "epoch19", "epoch20", "tail3_mean")))
            exact_tie_case = any(margins[source] == 0.0 for source in SOURCES)
            mixed_negative = (gold == "SUPPORT" and not refute_involved and
                predictions["tail3_mean"] == "NOT_ENTITLED" and not persistent and
                not exact_tie_case)
            output.append({"run": run, "training_seed": item["seed"], "split_seed": 174,
                "arm": item["arm"], "dev_position": position, "gold_label": gold,
                "selected_epoch": selected_epoch,
                **{f"{source}_logits": logits_by_source[source] for source in SOURCES},
                **{f"{source}_prediction": predictions[source] for source in SOURCES},
                **{f"{source}_entitlement_margin": margins[source] for source in SOURCES},
                **{f"{source}_refute_margin": refute_margins[source] for source in SOURCES},
                "late_entitlement_margin_population_stddev": population_stddev(late_margins),
                "late_entitlement_margin_range": max(late_margins) - min(late_margins),
                "consecutive_sign_change_count": sign_changes,
                "support_sign_vote_count": positive_votes,
                "not_entitled_sign_vote_count": negative_votes, "exact_zero_tie_count": zero_votes,
                "sign_pattern": "".join(signs),
                "selected_to_mean_transition": f"{predictions['selected']}->{predictions['tail3_mean']}",
                "selected_to_median_transition": f"{predictions['selected']}->{predictions['tail3_median']}",
                "refute_involved": refute_involved,
                "selected_consensus_outlier": selected_consensus,
                "mean_magnitude_override": magnitude_override, "median_rescue": median_rescue,
                "persistent_stable_negative": persistent,
                "temporally_mixed_negative": mixed_negative})
    if len(output) != 4320:
        raise RuntimeError("row decomposition must contain exactly 4320 rows")
    if [(row["run"], row["dev_position"]) for row in output] != [
            (run, position) for run in RUNS for position in range(720)]:
        raise RuntimeError("row decomposition order mismatch")
    return output


def scopes() -> list[tuple[str, str, tuple[str, ...]]]:
    result = [("run", run, (run,)) for run in RUNS]
    result.extend(("arm", arm, tuple(run for run in RUNS if run.endswith(f"_{arm}")))
                  for arm in ARMS)
    result.append(("pooled", "all_six_runs", RUNS))
    return result


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def support_decomposition(scope: str, aggregate_name: str, run_names: tuple[str, ...],
                          rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["run"] in run_names and row["gold_label"] == "SUPPORT"]
    count = len(selected)
    selected_tp = sum(row["selected_prediction"] == "SUPPORT" for row in selected)
    mean_tp = sum(row["tail3_mean_prediction"] == "SUPPORT" for row in selected)
    median_tp = sum(row["tail3_median_prediction"] == "SUPPORT" for row in selected)
    losses = [row for row in selected if row["selected_prediction"] == "SUPPORT" and
              row["tail3_mean_prediction"] != "SUPPORT"]
    gains = [row for row in selected if row["selected_prediction"] != "SUPPORT" and
             row["tail3_mean_prediction"] == "SUPPORT"]
    return {"scope": scope, "aggregate": aggregate_name, "run_count": len(run_names),
        "gold_support_count": count, "selected_support_true_positives": selected_tp,
        "tail3_mean_support_true_positives": mean_tp,
        "tail3_median_support_true_positives": median_tp,
        "selected_support_recall": ratio(selected_tp, count),
        "tail3_mean_support_recall": ratio(mean_tp, count),
        "tail3_median_support_recall": ratio(median_tp, count),
        "selected_to_mean_support_losses": len(losses),
        "selected_to_mean_support_gains": len(gains),
        "net_support_true_positive_change": mean_tp - selected_tp,
        "losses_to_NOT_ENTITLED": sum(row["tail3_mean_prediction"] == "NOT_ENTITLED" for row in losses),
        "losses_to_REFUTE": sum(row["tail3_mean_prediction"] == "REFUTE" for row in losses),
        "gains_from_NOT_ENTITLED": sum(row["selected_prediction"] == "NOT_ENTITLED" for row in gains),
        "gains_from_REFUTE": sum(row["selected_prediction"] == "REFUTE" for row in gains),
        "selected_consensus_outlier_count": sum(row["selected_consensus_outlier"] for row in selected),
        "mean_magnitude_override_count": sum(row["mean_magnitude_override"] for row in selected),
        "median_rescue_count": sum(row["median_rescue"] for row in selected),
        "persistent_stable_negative_count": sum(row["persistent_stable_negative"] for row in selected),
        "temporally_mixed_negative_count": sum(row["temporally_mixed_negative"] for row in selected),
        "tail3_mean_NOT_ENTITLED_false_negative_count": sum(
            row["tail3_mean_prediction"] == "NOT_ENTITLED" for row in selected),
        "tail3_mean_REFUTE_false_negative_count": sum(
            row["tail3_mean_prediction"] == "REFUTE" for row in selected),
        "refute_involved_gold_support_count": sum(row["refute_involved"] for row in selected)}


def build_aggregates(rows: list[dict[str, Any]], tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pooled_mechanism: dict[str, Any] | None = None
    for scope, aggregate_name, run_names in scopes():
        support = support_decomposition(scope, aggregate_name, run_names, rows)
        tables["support"].append(support)
        loss_denom = support["losses_to_NOT_ENTITLED"]
        magnitude_denom = loss_denom
        rescue_denom = support["mean_magnitude_override_count"]
        persistent_denom = support["tail3_mean_NOT_ENTITLED_false_negative_count"]
        scoped_support_rows = [row for row in rows if row["run"] in run_names and
                               row["gold_label"] == "SUPPORT"]
        mechanism = {"scope": scope, "aggregate": aggregate_name, "run_count": len(run_names),
            "selected_to_mean_support_losses_to_NOT_ENTITLED": loss_denom,
            "selected_consensus_outlier_count": support["selected_consensus_outlier_count"],
            "mean_magnitude_override_count": support["mean_magnitude_override_count"],
            "median_rescue_count": support["median_rescue_count"],
            "tail3_mean_NOT_ENTITLED_false_negative_count": persistent_denom,
            "persistent_stable_negative_count": support["persistent_stable_negative_count"],
            "temporally_mixed_negative_count": support["temporally_mixed_negative_count"],
            "consensus_outlier_denominator": loss_denom,
            "consensus_outlier_share": ratio(support["selected_consensus_outlier_count"], loss_denom),
            "magnitude_override_denominator": magnitude_denom,
            "magnitude_override_share": ratio(support["mean_magnitude_override_count"], magnitude_denom),
            "median_rescue_denominator": rescue_denom,
            "median_rescue_rate": ratio(support["median_rescue_count"], rescue_denom),
            "persistent_bias_denominator": persistent_denom,
            "persistent_bias_share": ratio(support["persistent_stable_negative_count"], persistent_denom),
            "refute_involved_support_loss_count": sum(
                row["selected_prediction"] == "SUPPORT" and
                row["tail3_mean_prediction"] != "SUPPORT" and row["refute_involved"]
                for row in scoped_support_rows)}
        tables["mechanisms"].append(mechanism)
        if scope == "pooled":
            pooled_mechanism = mechanism
        scoped_rows = [row for row in rows if row["run"] in run_names]
        for gold in LABELS:
            gold_rows = [row for row in scoped_rows if row["gold_label"] == gold]
            for source in SOURCES:
                margins = [row[f"{source}_entitlement_margin"] for row in gold_rows]
                predictions = [row[f"{source}_prediction"] for row in gold_rows]
                count = len(margins)
                if count == 0:
                    raise RuntimeError(f"{scope}/{aggregate_name}/{gold}: empty gold group")
                tables["gold"].append({"scope": scope, "aggregate": aggregate_name,
                    "run_count": len(run_names), "gold_label": gold, "source": source,
                    "count": count, "mean_entitlement_margin": mean64(margins),
                    "median_entitlement_margin": median64(margins),
                    "population_stddev": population_stddev(margins), "minimum": min(margins),
                    "maximum": max(margins),
                    "negative_margin_fraction": sum(value < 0.0 for value in margins) / count,
                    "positive_margin_fraction": sum(value > 0.0 for value in margins) / count,
                    "exact_zero_fraction": sum(value == 0.0 for value in margins) / count,
                    **{f"pred_{label}": predictions.count(label) for label in LABELS}})
            pattern_names = sorted({row["sign_pattern"] for row in gold_rows})
            for pattern in pattern_names:
                pattern_rows = [row for row in gold_rows if row["sign_pattern"] == pattern]
                mean_transitions = {transition: sum(
                    row["selected_to_mean_transition"] == transition for row in pattern_rows)
                    for transition in sorted({row["selected_to_mean_transition"] for row in pattern_rows})}
                median_transitions = {transition: sum(
                    row["selected_to_median_transition"] == transition for row in pattern_rows)
                    for transition in sorted({row["selected_to_median_transition"] for row in pattern_rows})}
                tables["patterns"].append({"scope": scope, "aggregate": aggregate_name,
                    "run_count": len(run_names), "gold_label": gold, "sign_pattern": pattern,
                    "count": len(pattern_rows), "fraction": len(pattern_rows) / len(gold_rows),
                    "mean_margin_range": mean64([row["late_entitlement_margin_range"]
                                                  for row in pattern_rows]),
                    "mean_margin_population_stddev": mean64([
                        row["late_entitlement_margin_population_stddev"] for row in pattern_rows]),
                    "selected_to_mean_transition_counts": mean_transitions,
                    "selected_to_median_transition_counts": median_transitions})
        mean_to_median_gains = [row for row in scoped_support_rows
            if row["tail3_mean_prediction"] != "SUPPORT" and row["tail3_median_prediction"] == "SUPPORT"]
        mean_to_median_losses = [row for row in scoped_support_rows
            if row["tail3_mean_prediction"] == "SUPPORT" and row["tail3_median_prediction"] != "SUPPORT"]
        mean_tp = support["tail3_mean_support_true_positives"]
        median_tp = support["tail3_median_support_true_positives"]
        tables["counterfactual"].append({"scope": scope, "aggregate": aggregate_name,
            "run_count": len(run_names), "gold_support_count": support["gold_support_count"],
            "tail3_mean_support_true_positives": mean_tp,
            "tail3_median_support_true_positives": median_tp,
            "tail3_mean_support_recall": support["tail3_mean_support_recall"],
            "tail3_median_support_recall": support["tail3_median_support_recall"],
            "mean_to_median_support_gains": len(mean_to_median_gains),
            "mean_to_median_support_losses": len(mean_to_median_losses),
            "mean_NOT_ENTITLED_to_median_SUPPORT": sum(
                row["tail3_mean_prediction"] == "NOT_ENTITLED" for row in mean_to_median_gains),
            "mean_REFUTE_to_median_SUPPORT": sum(
                row["tail3_mean_prediction"] == "REFUTE" for row in mean_to_median_gains),
            "mean_SUPPORT_to_median_NOT_ENTITLED": sum(
                row["tail3_median_prediction"] == "NOT_ENTITLED" for row in mean_to_median_losses),
            "mean_SUPPORT_to_median_REFUTE": sum(
                row["tail3_median_prediction"] == "REFUTE" for row in mean_to_median_losses),
            "mean_magnitude_override_count": support["mean_magnitude_override_count"],
            "median_rescue_count": support["median_rescue_count"],
            "median_rescue_rate": ratio(support["median_rescue_count"],
                                        support["mean_magnitude_override_count"]),
            "diagnostic_only": True})
    if pooled_mechanism is None:
        raise RuntimeError("pooled mechanism summary was not created")
    return pooled_mechanism


def apply_decision(pooled: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    losses = pooled["selected_to_mean_support_losses_to_NOT_ENTITLED"]
    consensus_share = pooled["consensus_outlier_share"]
    magnitude_share = pooled["magnitude_override_share"]
    magnitude_count = pooled["mean_magnitude_override_count"]
    rescue_rate = pooled["median_rescue_rate"]
    mean_ne_fn = pooled["tail3_mean_NOT_ENTITLED_false_negative_count"]
    persistent_count = pooled["persistent_stable_negative_count"]
    persistent_share = pooled["persistent_bias_share"]

    def threshold(mechanism: str, level: str, name: str, required: Any,
                  observed: Any, passed: bool) -> bool:
        tables["criteria"].append({"mechanism": mechanism, "evidence_level": level,
            "criterion": name, "required": required, "observed": observed, "passed": passed})
        return passed

    temporal_strong_parts = [
        threshold("temporal_consensus", "strong", "losses_to_not_entitled_minimum", ">=12", losses, losses >= 12),
        threshold("temporal_consensus", "strong", "consensus_outlier_share_minimum", ">=0.60",
                  consensus_share, consensus_share is not None and consensus_share >= 0.60),
        threshold("temporal_consensus", "strong", "magnitude_override_share_maximum", "<=0.25",
                  magnitude_share, magnitude_share is not None and magnitude_share <= 0.25),
    ]
    temporal_moderate_parts = [threshold("temporal_consensus", "moderate",
        "consensus_outlier_share_minimum", ">=0.40", consensus_share,
        consensus_share is not None and consensus_share >= 0.40)]
    magnitude_strong_parts = [
        threshold("mean_magnitude", "strong", "losses_to_not_entitled_minimum", ">=12", losses, losses >= 12),
        threshold("mean_magnitude", "strong", "magnitude_override_share_minimum", ">=0.40",
                  magnitude_share, magnitude_share is not None and magnitude_share >= 0.40),
        threshold("mean_magnitude", "strong", "mean_magnitude_override_count_minimum", ">=8",
                  magnitude_count, magnitude_count >= 8),
        threshold("mean_magnitude", "strong", "median_rescue_rate_minimum", ">=0.70",
                  rescue_rate, rescue_rate is not None and rescue_rate >= 0.70),
    ]
    magnitude_moderate_parts = [
        threshold("mean_magnitude", "moderate", "magnitude_override_share_minimum", ">=0.25",
                  magnitude_share, magnitude_share is not None and magnitude_share >= 0.25),
        threshold("mean_magnitude", "moderate", "median_rescue_rate_minimum", ">=0.50",
                  rescue_rate, rescue_rate is not None and rescue_rate >= 0.50),
    ]
    persistent_strong_parts = [
        threshold("persistent_boundary_bias", "strong", "mean_not_entitled_false_negatives_minimum",
                  ">=30", mean_ne_fn, mean_ne_fn >= 30),
        threshold("persistent_boundary_bias", "strong", "persistent_stable_negative_count_minimum",
                  ">=20", persistent_count, persistent_count >= 20),
        threshold("persistent_boundary_bias", "strong", "persistent_bias_share_minimum", ">=0.60",
                  persistent_share, persistent_share is not None and persistent_share >= 0.60),
    ]
    persistent_moderate_parts = [threshold("persistent_boundary_bias", "moderate",
        "persistent_bias_share_minimum", ">=0.40", persistent_share,
        persistent_share is not None and persistent_share >= 0.40)]
    strong = {"temporal_consensus": all(temporal_strong_parts),
              "mean_magnitude": all(magnitude_strong_parts),
              "persistent_boundary_bias": all(persistent_strong_parts)}
    moderate = {"temporal_consensus": all(temporal_moderate_parts),
                "mean_magnitude": all(magnitude_moderate_parts),
                "persistent_boundary_bias": all(persistent_moderate_parts)}
    temporal_only = strong["temporal_consensus"] and all(
        not strong[name] and not moderate[name]
        for name in ("mean_magnitude", "persistent_boundary_bias"))
    magnitude_only = strong["mean_magnitude"] and all(
        not strong[name] and not moderate[name]
        for name in ("temporal_consensus", "persistent_boundary_bias"))
    persistent_only = strong["persistent_boundary_bias"] and all(
        not strong[name] and not moderate[name]
        for name in ("temporal_consensus", "mean_magnitude"))
    strong_names = [name for name, passed in strong.items() if passed]
    moderate_names = [name for name, passed in moderate.items() if passed]
    mixed = (len(strong_names) >= 2 or
             (len(strong_names) == 1 and any(name not in strong_names for name in moderate_names)) or
             (len(strong_names) == 0 and len(moderate_names) >= 2))
    if temporal_only:
        decision = TEMPORAL_DOMINANT
    elif magnitude_only:
        decision = MAGNITUDE_DOMINANT
    elif persistent_only:
        decision = PERSISTENT_DOMINANT
    elif mixed:
        decision = MIXED
    else:
        decision = INCONCLUSIVE
    conditions = {BLOCKED: False, TEMPORAL_DOMINANT: temporal_only,
                  MAGNITUDE_DOMINANT: magnitude_only, PERSISTENT_DOMINANT: persistent_only,
                  MIXED: mixed, INCONCLUSIVE: not any(
                      (temporal_only, magnitude_only, persistent_only, mixed))}
    taxonomy_text = {
        BLOCKED: "fail-closed exception only",
        TEMPORAL_DOMINANT: "strong temporal consensus and no other strong or moderate evidence",
        MAGNITUDE_DOMINANT: "strong mean magnitude and no other strong or moderate evidence",
        PERSISTENT_DOMINANT: "strong persistent bias and no other strong or moderate evidence",
        MIXED: "precommitted multi-mechanism strong/moderate conjunction",
        INCONCLUSIVE: "integrity passes and no positive localization condition applies",
    }
    for alternative in DECISIONS:
        tables["decision"].append({"decision": alternative,
            "taxonomy_condition": taxonomy_text[alternative], "required": alternative == decision,
            "observed": conditions[alternative],
            "passed": (alternative == decision) == conditions[alternative]})
    if sum(row["required"] is True for row in tables["decision"]) != 1 or not all(
            row["passed"] for row in tables["decision"]):
        raise RuntimeError("Stage194-A decision taxonomy closure failed")
    return decision, {"strong_evidence": strong, "moderate_evidence": moderate,
        "selected_decision": decision, "conditions": conditions,
        "primary_pooled_ratios": {
            "consensus_outlier_share": consensus_share,
            "magnitude_override_share": magnitude_share,
            "median_rescue_rate": rescue_rate,
            "persistent_bias_share": persistent_share,
        }, "primary_pooled_denominators": {
            "consensus_outlier_denominator": pooled["consensus_outlier_denominator"],
            "magnitude_override_denominator": pooled["magnitude_override_denominator"],
            "median_rescue_denominator": pooled["median_rescue_denominator"],
            "persistent_bias_denominator": pooled["persistent_bias_denominator"],
        }}


def recommendation(decision: str, pooled: dict[str, Any]) -> str:
    if decision == TEMPORAL_DOMINANT:
        return ("Design one interpretable EMA/SWA-style temporal-consensus mechanism, "
                "but do not authorize it.")
    if decision == MAGNITUDE_DOMINANT:
        return ("Design a separate robust-aggregation diagnostic such as median or trimmed mean; "
                "do not authorize EMA/SWA training.")
    if decision == PERSISTENT_DOMINANT:
        return ("Design an explicit mechanistically interpretable entitlement-boundary mechanism "
                "or calibration; authorize neither.")
    if decision == MIXED:
        shares = {"temporal_consensus": pooled["consensus_outlier_share"],
                  "mean_magnitude": pooled["magnitude_override_share"],
                  "persistent_boundary_bias": pooled["persistent_bias_share"]}
        available = {name: value for name, value in shares.items() if value is not None}
        largest = max(available, key=available.get) if available else "no ratio with a nonzero denominator"
        return ("Do not combine fixes. Design and test one mechanism at a time, beginning with "
                f"{largest}, the largest available evidence share.")
    return "Integrity passed, but the precommitted mechanism evidence is inconclusive; authorize no intervention."


def analyze(args: argparse.Namespace, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    repo, stage193a, run_root, stage193c, _ = establish_safe_paths(args)
    source_identity = validate_source_identity(repo, args.current_diagnostic_git_commit, tables["closure"])
    stage193a_report, manifests = validate_stage193a(stage193a, run_root, tables["closure"])
    stage193c_report = validate_stage193c(stage193c, stage193a, run_root, tables["closure"])
    data = validate_runs(manifests, run_root, tables["identity"])
    tables["rows"].extend(decompose_rows(data))
    pooled = build_aggregates(tables["rows"], tables)
    decision, taxonomy = apply_decision(pooled, tables)
    restrictions = ["diagnostic only", "artifact-only analysis",
        "no checkpoint/model/capsule loading", "no external data",
        "median aggregation is diagnostic only", "no statistical-significance claim",
        "no model advancement", "no training authorization", "no EMA/SWA authorization",
        "no calibration authorization"]
    return {"stage": "Stage194-A", "decision": decision, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True, "artifact_only_analysis": True,
        "checkpoint_loaded": False, "model_loaded": False, "capsule_loaded": False,
        "external_data_used": False, "median_aggregation_diagnostic_only": True,
        "statistical_significance_claim": False, "model_advancement_decision": False,
        "training_authorized": False, "subsequent_training_authorized": False,
        "ema_authorized": False, "swa_authorized": False, "calibration_authorized": False,
        "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
        "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_SHA256,
        "canonical_labels": list(LABELS), "ordered_runs": list(RUNS),
        "stage193a_directory": str(stage193a), "stage193b_run_root": str(run_root),
        "stage193c_directory": str(stage193c), "stage193a_decision": stage193a_report["decision"],
        "stage193c_decision": stage193c_report["decision"],
        "diagnostic_source_identity": source_identity,
        "loaded_prediction_epochs_for_calculation": {
            run: [data[run]["selected_epoch"], 18, 19, 20] for run in RUNS},
        "all_twenty_exports_validated_per_run": True,
        "row_margin_decomposition_count": len(tables["rows"]),
        "primary_pooled_mechanism_summary": pooled,
        "decision_taxonomy": taxonomy, "recommended_next_stage": recommendation(decision, pooled),
        "interpretation_restrictions": restrictions, "exception": None}


def blocked_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {"stage": "Stage194-A", "decision": BLOCKED, "runnable": False,
        "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "diagnostic_only": True,
        "artifact_only_analysis": True, "checkpoint_loaded": False, "model_loaded": False,
        "capsule_loaded": False, "external_data_used": False,
        "median_aggregation_diagnostic_only": True, "statistical_significance_claim": False,
        "model_advancement_decision": False, "training_authorized": False,
        "subsequent_training_authorized": False, "ema_authorized": False,
        "swa_authorized": False, "calibration_authorized": False,
        "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
        "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": TRAINER_SHA256,
        "recommended_next_stage": "Resolve the frozen-input or analysis failure; authorize no intervention.",
        "interpretation_restrictions": ["diagnostic only", "artifact-only analysis",
            "no checkpoint/model/capsule loading", "no external data",
            "median aggregation is diagnostic only", "no statistical-significance claim",
            "no model advancement", "no training authorization", "no EMA/SWA authorization",
            "no calibration authorization"],
        "exception": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()}}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Stage194-A entitlement-boundary localization report", "",
        f"Decision: `{report['decision']}`", "", f"- Runnable: {str(report['runnable']).lower()}",
        "- Diagnostic only: true", "- Artifact-only analysis: true",
        "- Checkpoint/model/capsule loading: none", "- External data used: false",
        "- Median aggregation: diagnostic counterfactual only",
        "- Statistical-significance claim: false", "- Model advancement: false",
        "- Training authorized: false", "- EMA authorized: false", "- SWA authorized: false",
        "- Calibration authorized: false", "",
        f"- Stage193 runtime repository commit: `{report.get('stage193_runtime_repository_commit')}`",
        f"- Trainer blob commit: `{report.get('trainer_blob_commit')}`",
        f"- Trainer SHA256: `{report.get('trainer_blob_sha256')}`", "", "## Interpretation", ""]
    if report["decision"] == BLOCKED:
        lines.append("Localization is blocked by a provenance, frozen-input, schema, hash, cardinality, alignment, or analysis failure.")
    elif report["decision"] == TEMPORAL_DOMINANT:
        lines.append("The precommitted evidence localizes the primary behavior to a selected-checkpoint temporal-consensus outlier.")
    elif report["decision"] == MAGNITUDE_DOMINANT:
        lines.append("The precommitted evidence localizes the primary behavior to mean-logit magnitude domination.")
    elif report["decision"] == PERSISTENT_DOMINANT:
        lines.append("The precommitted evidence localizes the primary behavior to persistent entitlement-boundary bias.")
    elif report["decision"] == MIXED:
        lines.append("The precommitted evidence supports multiple temporal and/or boundary mechanisms.")
    else:
        lines.append("Integrity passed, but the precommitted localization evidence is inconclusive.")
    lines.extend(["", "## Recommendation", "", report["recommended_next_stage"], "",
        "This decision authorizes no production inference rule, training, model advancement, EMA, SWA, or calibration.", ""])
    return "\n".join(lines)


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
        report = analyze(args, tables)
    except BaseException as exc:
        report = blocked_report(args, exc)
        tables["decision"].append({"decision": BLOCKED,
            "taxonomy_condition": "fail-closed exception only", "required": True,
            "observed": {"type": type(exc).__name__, "message": str(exc)}, "passed": True})
    write_json(output / OUTPUTS["json"], report)
    (output / OUTPUTS["md"]).write_text(markdown(report), encoding="utf-8")
    for name, header in CSV_HEADERS.items():
        write_csv(output / OUTPUTS[name], header, tables[name])
    return 0 if report["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
