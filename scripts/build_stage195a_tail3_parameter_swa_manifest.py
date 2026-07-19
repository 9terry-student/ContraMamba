#!/usr/bin/env python3
"""Build the fail-closed Stage195-A tail-three parameter-SWA manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

TRAINER_BLOB_COMMIT = "bd27e46daf218a57da9a3142c9e4bc5cc44ad53a"
TRAINER_BLOB_SHA256 = "4fe903c9f3aa21ee6365a0297c27e4a333d295dbb851384efc7bc8d3f7607954"
STAGE195P0_SPEC_SHA256 = "a65eab7877c3768e545fed070932b432bd0459374386522051d75af1d5254a60"
STAGE193_RUNTIME_COMMIT = "89a9805d0e9c877774f9ce4b356297d31645b74b"
STAGE193_TRAINER_BLOB_COMMIT = "e83d8af756fa84b7a91c14e0910ae388b07b5f02"
STAGE193_TRAINER_BLOB_SHA256 = "25d42bdcd204219a2b2e5e7bf2a8b14459eafb4945c05c61ab3611bc9e7365bc"
SIDECAR_SHA256 = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
STAGE185_BASENAME = "stage185a_controlled_train_integrity_sidecar_20260715_141914"
READY = "STAGE195A_TAIL3_PARAMETER_SWA_MANIFEST_READY"
BLOCKED = "STAGE195A_TAIL3_PARAMETER_SWA_MANIFEST_BLOCKED"
STAGE193_READY = "STAGE193A_TAIL3_FRESH_SEED_MANIFEST_READY"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEEDS = (180, 181, 182)
SOURCE_SEEDS = (177, 178, 179)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
SOURCE_RUNS = tuple(f"seed{seed}_{arm}" for seed in SOURCE_SEEDS for arm in ARMS)

OUTPUT_OPTIONS = {
    "--output-json": "training_report.json",
    "--output-predictions-json": "clean_dev_predictions.json",
    "--stage115-clean-dev-scalar-output-jsonl": "clean_dev_scalars.jsonl",
}
SOURCE_OBSERVABILITY_FLAGS = (
    "--stage193-tail3-fresh-seed-observability",
    "--stage191-trajectory-replay-observability",
    "--stage191-save-trajectory-state-capsules",
)
STAGE195_FLAG = "--stage195-tail3-parameter-swa-causal-test"
STAGE195_OUTPUT_OPTION = "--stage195-tail3-parameter-swa-output-dir"
ARM_OPTIONS = (
    "--compatible-positive-margin-weight",
    "--controlled-integrity-sidecar-path",
    "--expected-integrity-sidecar-semantic-sha256",
)

FORBIDDEN_PATH_OPTIONS = (
    "ood_data", "output_ood_json", "output_ood_predictions_json", "external_data",
    "external_output_dir", "external_eval_jsonl", "external_eval_name",
    "stage43_external_factver_jsonl", "stage57_bridge_train_jsonl",
    "stage66_bridge_train_jsonl", "stage75_bridge_train_jsonl",
    "stage80a_bridge_train_jsonl", "pair_contrastive_frame_data",
    "temporal_diagnostic_data", "v7_temporal_safety_data",
    "v7_temporal_mismatch_multihead_data", "v7_temporal_preservation_data",
    "v7_coverage_entailment_data",
)
FORBIDDEN_FLAGS = (
    "smoke", "loss_sweep", "enable_external_eval", "enable_stage43_external_eval",
    "stage43_external_enable_shadow_export", "use_pair_contrastive_frame_loss",
    "use_temporal_diagnostic_loss", "use_td_constrained_selection",
    "use_temporal_residual_adapter", "use_temporal_adapter_loss",
    "use_temporal_adapter_final_penalty", "use_temporal_channel",
    "use_temporal_channel_loss", "use_temporal_channel_gated_penalty",
    "v7_use_temporal_safety_head", "v7_use_temporal_safety_loss",
    "v7_use_temporal_mismatch_multihead", "v7_use_temporal_mismatch_multihead_loss",
    "v7_use_temporal_preservation_head", "v7_use_temporal_preservation_loss",
    "v7_use_temporal_preservation_aware_cap", "v7_use_coverage_entailment_head",
    "v7_use_coverage_entailment_loss", "use_preservation_constrained_selection",
    "stage44_use_anti_collapse_selection",
)
FORBIDDEN_MODES = (
    "stage57_bridge_train_mode", "stage66_bridge_train_mode",
    "stage75_bridge_train_mode", "stage80a_bridge_train_mode",
    "v7_temporal_safety_cap_mode", "v7_temporal_mismatch_multihead_cap_mode",
)
FORBIDDEN_CALIBRATION_OPTIONS = (
    "ood_unflagged_ne_shift_sweep", "ood_selective_ne_shift_sweep",
    "dev_calibrated_ne_shift_candidates", "dev_calibrated_ne_frame_penalty_candidates",
)

STAGE193_OUTPUTS = {
    "stage193a_tail3_fresh_seed_manifest_report.json",
    "stage193a_tail3_fresh_seed_manifest_report.md",
    "stage193a_run_manifest.jsonl",
    "stage193a_run_command_matrix.csv",
    "stage193a_source_and_template_gate.csv",
    "stage193a_precommitted_gate.csv",
}
OUTPUTS = {
    "json": "stage195a_tail3_parameter_swa_manifest.json",
    "md": "stage195a_tail3_parameter_swa_manifest.md",
    "jsonl": "stage195a_run_manifest.jsonl",
    "matrix": "stage195a_run_command_matrix.csv",
    "source": "stage195a_source_and_template_gate.csv",
    "gates": "stage195a_precommitted_gate.csv",
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
GATE_HEADER = ["gate", "required", "observed", "passed", "blocking_reason"]
SOURCE_MATRIX_HEADER = [
    "run", "training_seed", "split_seed", "arm", "planned_run_directory",
    "planned_output_json_path", "planned_selected_checkpoint_path",
    "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
    "runtime_repository_commit", "command", "expected_trajectory_rows",
    "expected_prediction_exports", "expected_prediction_rows_per_export",
    "expected_state_capsules",
]

REPORT_KEYS = {
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
ROW_KEYS = {
    "stage", "run", "training_seed", "split_seed", "arm", "canonical_labels",
    "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
    "stage195_runtime_repository_commit", "argv", "command_argv", "command",
    "planned_run_directory", "planned_output_json_path",
    "planned_selected_checkpoint_path", "expected_trajectory_contract_path",
    "expected_trajectory_ledger_path", "expected_prediction_export_paths",
    "expected_stage195_swa_predictions_path", "expected_stage195_swa_metrics_path",
    "expected_stage195_swa_contract_path", "expected_trajectory_rows",
    "expected_prediction_exports", "expected_prediction_rows_per_export",
    "expected_stage195_swa_prediction_rows", "expected_state_capsules",
    "expected_swa_checkpoints", "logits_source", "arm_contract",
    "argv_mutation_audit", "frozen_training_envelope", "runnable", "diagnostic_only",
    "exact_six_run_diagnostic_execution_authorized", "model_advancement_authorized",
    "production_swa_selected", "entitlement_correction_implemented",
    "stage195c_decision_made", "subsequent_training_authorized",
    "statistical_significance_claimed", "external_data_used",
}
STAGE193_REPORT_KEYS = {
    "stage", "decision", "runnable", "blocking_reasons", "diagnostic_only",
    "exact_six_run_diagnostic_execution_authorized",
    "training_for_model_advancement_authorized", "model_advancement_decision",
    "subsequent_training_authorized", "external_data_used", "checkpoint_loaded",
    "model_loaded", "capsule_loaded", "statistical_significance_claim",
    "current_diagnostic_git_commit", "source_identity", "trainer_blob_commit",
    "trainer_blob_sha256", "stage193_runtime_repository_commit",
    "frozen_source_identities", "stage193b_run_root", "ordered_runs",
    "run_manifest_count", "expected_trajectory_rows_per_run",
    "expected_prediction_exports_per_run", "expected_prediction_rows_per_export",
    "expected_state_capsules_per_run", "canonical_labels", "logits_source",
    "source_and_template_gates", "precommitted_gates", "exception",
}
STAGE193_ROW_KEYS = {
    "stage", "run", "training_seed", "split_seed", "arm", "canonical_labels",
    "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
    "stage193_runtime_repository_commit", "argv", "command_argv", "command",
    "planned_run_directory", "planned_output_json_path",
    "planned_selected_checkpoint_path", "expected_trajectory_contract_path",
    "expected_trajectory_ledger_path", "expected_prediction_export_paths",
    "expected_trajectory_rows", "expected_prediction_exports",
    "expected_prediction_rows_per_export", "expected_state_capsules", "logits_source",
    "arm_contract", "argv_mutation_audit", "frozen_training_envelope", "runnable",
    "diagnostic_only", "exact_six_run_diagnostic_execution_authorized",
    "training_for_model_advancement_authorized", "model_advancement_decision",
    "subsequent_training_authorized", "external_data_used",
}
STAGE193_SOURCE_GATE_NAMES = (
    "diagnostic_and_trainer_source_identity", "stage192a_frozen_closure",
    "stage185_semantic_sha256", "stage191b_frozen_closure",
    "stage191b_ordered_source_matrix", "baseline_three_source_template_equivalence",
    "intervention_three_source_template_equivalence",
)
STAGE193_PRECOMMITTED_GATE_NAMES = (
    "manifest_strict_non_bool_integer_contract", "report_strict_non_bool_integer_contract",
    "exact_fresh_run_order", "stage193b_run_root_empty_or_absent",
    "diagnostic_authorization_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage193a-dir", type=Path, required=True)
    parser.add_argument("--stage185-sidecar-dir", type=Path, required=True)
    parser.add_argument("--stage195b-run-root", type=Path, required=True)
    parser.add_argument("--current-diagnostic-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def exact_int(value: Any) -> bool:
    return type(value) is int


def ensure_exact_keys(value: Any, keys: set[str], context: str) -> None:
    if type(value) is not dict or set(value) != keys:
        observed = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{context}: exact key closure mismatch; observed {observed!r}")


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
                raise ValueError(f"{path}:{number}: JSONL row is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ValueError(f"{path}: fixed CSV header mismatch")
        rows = list(reader)
    if any(set(row) != set(expected_header) or None in row for row in rows):
        raise ValueError(f"{path}: CSV row closure mismatch")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sidecar_sha(path: Path) -> tuple[str, int]:
    rows = read_jsonl(path)
    canonical = [{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows)


def git_call(repo: Path, arguments: list[str], *, binary: bool = False, dirty: bool = False) -> Any:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=False, capture_output=True, shell=False
    )
    if dirty:
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def source_identity(repo: Path, commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        raise ValueError("current diagnostic commit must be lowercase hexadecimal length 40")
    head = git_call(repo, ["rev-parse", "HEAD"])
    files: dict[str, Any] = {}
    for relative in (
        "reports/stage195a_tail3_parameter_swa_manifest_spec.md",
        "scripts/build_stage195a_tail3_parameter_swa_manifest.py",
    ):
        current = (repo / relative).read_bytes()
        blob = git_call(repo, ["show", f"{commit}:{relative}"], binary=True)
        files[relative] = {
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "bytes_equal": current == blob,
            "unstaged_clean": git_call(repo, ["diff", "--quiet", "--", relative], dirty=True) == 0,
            "staged_clean": git_call(repo, ["diff", "--cached", "--quiet", "--", relative], dirty=True) == 0,
        }

    trainer_relative = "scripts/train_controlled_v6b_minimal.py"
    trainer_current = (repo / trainer_relative).read_bytes()
    trainer_blob = git_call(
        repo, ["show", f"{TRAINER_BLOB_COMMIT}:{trainer_relative}"], binary=True
    )
    trainer_current_sha = hashlib.sha256(trainer_current).hexdigest()
    trainer_blob_sha = hashlib.sha256(trainer_blob).hexdigest()
    trainer = {
        "path": str((repo / trainer_relative).resolve()),
        "blob_commit": TRAINER_BLOB_COMMIT,
        "frozen_sha256": TRAINER_BLOB_SHA256,
        "current_sha256": trainer_current_sha,
        "commit_blob_sha256": trainer_blob_sha,
        "bytes_equal": trainer_current == trainer_blob,
        "sha256_equal": trainer_current_sha == trainer_blob_sha == TRAINER_BLOB_SHA256,
    }

    spec_relative = "reports/stage195p0_tail3_parameter_swa_spec.md"
    spec_current = (repo / spec_relative).read_bytes()
    spec_blob = git_call(
        repo, ["show", f"{TRAINER_BLOB_COMMIT}:{spec_relative}"], binary=True
    )
    spec_current_sha = hashlib.sha256(spec_current).hexdigest()
    spec_blob_sha = hashlib.sha256(spec_blob).hexdigest()
    stage195p0_spec = {
        "path": str((repo / spec_relative).resolve()),
        "blob_commit": TRAINER_BLOB_COMMIT,
        "frozen_sha256": STAGE195P0_SPEC_SHA256,
        "current_sha256": spec_current_sha,
        "commit_blob_sha256": spec_blob_sha,
        "bytes_equal": spec_current == spec_blob,
        "sha256_equal": spec_current_sha == spec_blob_sha == STAGE195P0_SPEC_SHA256,
    }
    passed = (
        head == commit
        and trainer["bytes_equal"]
        and trainer["sha256_equal"]
        and stage195p0_spec["bytes_equal"]
        and stage195p0_spec["sha256_equal"]
        and all(
            item["bytes_equal"] and item["unstaged_clean"] and item["staged_clean"]
            for item in files.values()
        )
    )
    return {
        "supplied_commit": commit,
        "repository_head": head,
        "stage195a_files": files,
        "trainer": trainer,
        "stage195p0_specification": stage195p0_spec,
        "passed": passed,
    }


def establish_safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError("repo root is not a Git worktree directory")
    reports = (repo / "reports").resolve()
    if not reports.is_dir():
        raise ValueError("reports directory is absent")
    stage193 = args.stage193a_dir.resolve()
    stage185 = args.stage185_sidecar_dir.resolve()
    run_root = args.stage195b_run_root.resolve()
    output = args.output_dir.resolve()
    if (
        stage193.parent != reports
        or not stage193.name.startswith("stage193a_tail3_fresh_seed_manifest_")
        or not stage193.is_dir()
    ):
        raise ValueError("Stage193-A must be an explicit existing immediate reports child")
    if stage185 != (reports / STAGE185_BASENAME).resolve() or not stage185.is_dir():
        raise ValueError("Stage185 sidecar directory is not the exact authoritative directory")
    if output.parent != reports or not output.name.startswith(
        "stage195a_tail3_parameter_swa_manifest_"
    ):
        raise ValueError("Stage195-A output path is unsafe")
    if run_root.parent != reports or not run_root.name.startswith(
        "stage195b_tail3_parameter_swa_runs_"
    ):
        raise ValueError("Stage195-B planned run root is unsafe")
    if len({stage193, stage185, run_root, output}) != 4:
        raise ValueError("inputs, Stage195-B run root, and Stage195-A output must differ")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Stage195-A output exists and is not an empty directory")
    if run_root.exists() and (not run_root.is_dir() or any(run_root.iterdir())):
        raise ValueError("Stage195-B planned run root exists and is not empty")
    return repo, stage193, stage185, run_root, output


def option_map(argv: Any) -> dict[str, Any]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a list of strings")
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            raise ValueError(f"unsupported argv token: {token!r}")
        key = token[2:].replace("-", "_")
        if key in result:
            raise ValueError(f"duplicate argv option: {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[key] = argv[index + 1]
            index += 2
        else:
            result[key] = True
            index += 1
    return result


def normalized_template(argv: list[str], *, normalize_arm: bool = False) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in SOURCE_OBSERVABILITY_FLAGS:
            index += 1
            continue
        if token in (STAGE195_FLAG, STAGE195_OUTPUT_OPTION):
            raise ValueError("Stage193-A source argv contains a Stage195 option")
        if token == "--seed":
            if index + 1 >= len(argv):
                raise ValueError("source --seed value is missing")
            normalized.extend((token, "<TRAINING_SEED>"))
            index += 2
            continue
        if token in OUTPUT_OPTIONS:
            if index + 1 >= len(argv):
                raise ValueError(f"source {token} value is missing")
            normalized.extend((token, f"<PATH:{token}>"))
            index += 2
            continue
        if normalize_arm and token == "--compatible-positive-margin-weight":
            if index + 1 >= len(argv):
                raise ValueError("source margin-weight value is missing")
            normalized.extend((token, "<ARM_WEIGHT>"))
            index += 2
            continue
        if normalize_arm and token in ARM_OPTIONS[1:]:
            if index + 1 >= len(argv):
                raise ValueError(f"source {token} value is missing")
            index += 2
            continue
        normalized.append(token)
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            normalized.append(argv[index + 1])
            index += 2
        else:
            index += 1
    return normalized


def rewrite_argv(source: list[str], seed: int, run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    if source.count("--stage193-tail3-fresh-seed-observability") != 1:
        raise ValueError("source argv must contain Stage193 observability exactly once")
    if any(source.count(flag) != 0 for flag in SOURCE_OBSERVABILITY_FLAGS[1:]):
        raise ValueError("source Stage193 argv unexpectedly contains a Stage191 flag")
    if source.count("--seed") != 1:
        raise ValueError("source argv must contain --seed exactly once")
    if source.count(STAGE195_FLAG) or source.count(STAGE195_OUTPUT_OPTION):
        raise ValueError("source argv already contains Stage195 options")
    rewritten: list[str] = []
    changes: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    index = 0
    while index < len(source):
        token = source[index]
        if token in SOURCE_OBSERVABILITY_FLAGS:
            changes.append({"kind": "remove_flag", "token": token, "source_index": index})
            index += 1
            continue
        if token == "--seed":
            rewritten.extend((token, str(seed)))
            changes.append({"kind": "training_seed", "from": source[index + 1], "to": str(seed)})
            index += 2
            continue
        if token in OUTPUT_OPTIONS:
            if index + 1 >= len(source):
                raise ValueError(f"source {token} value is missing")
            replacement = str((run_dir / OUTPUT_OPTIONS[token]).resolve())
            rewritten.extend((token, replacement))
            seen_outputs.add(token)
            changes.append(
                {"kind": "output_path", "option": token, "from": source[index + 1], "to": replacement}
            )
            index += 2
            continue
        rewritten.append(token)
        if index + 1 < len(source) and not source[index + 1].startswith("--"):
            rewritten.append(source[index + 1])
            index += 2
        else:
            index += 1
    if seen_outputs != set(OUTPUT_OPTIONS):
        raise ValueError("source argv output option set is not exact")
    rewritten.append(STAGE195_FLAG)
    changes.append({"kind": "add_flag", "token": STAGE195_FLAG})
    rewritten.extend((STAGE195_OUTPUT_OPTION, str(run_dir)))
    changes.append(
        {"kind": "add_output_directory", "option": STAGE195_OUTPUT_OPTION, "to": str(run_dir)}
    )
    return rewritten, {"allowed_changes": changes, "all_other_tokens_and_order_preserved": True}


def exact_float_option(options: dict[str, Any], key: str, expected: float) -> bool:
    value = options.get(key)
    if type(value) is not str:
        return False
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed) and parsed == expected


def validate_envelope(
    argv: list[str], arm: str, seed: int, sidecar: Path, run_dir: Path
) -> dict[str, Any]:
    options = option_map(argv)
    required = {
        "architecture": "v6b_minimal", "backbone": "mamba", "device": "cuda",
        "model_name": "state-spaces/mamba-130m-hf", "flag_source": "controlled_heuristic",
        "data": "data/controlled_v5_v3_without_time_swap.jsonl", "seed": str(seed),
        "split_seed": "174", "epochs": "20", "select_metric": "final_macro_f1",
        "stage195_tail3_parameter_swa_causal_test": True,
        "stage195_tail3_parameter_swa_output_dir": str(run_dir),
    }
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError(f"{seed}/{arm}: frozen Stage195 runtime envelope mismatch")
    if not exact_float_option(options, "compatible_positive_margin_logit", 0.0):
        raise ValueError(f"{seed}/{arm}: compatible-positive margin logit is not 0.0")
    if options.get("use_temporal_comparator", True) is not True:
        raise ValueError(f"{seed}/{arm}: temporal comparator was disabled")
    if any(flag[2:].replace("-", "_") in options for flag in SOURCE_OBSERVABILITY_FLAGS):
        raise ValueError(f"{seed}/{arm}: source observability flag remains")
    if options.get("max_train_records") is not None:
        raise ValueError(f"{seed}/{arm}: max-train-record truncation is configured")
    for key in FORBIDDEN_PATH_OPTIONS:
        if options.get(key) not in (None, [], ""):
            raise ValueError(f"{seed}/{arm}: forbidden path/data option {key} is configured")
    if any(options.get(key) is True for key in FORBIDDEN_FLAGS):
        raise ValueError(f"{seed}/{arm}: a forbidden auxiliary/evaluation flag is enabled")
    if any(options.get(key, "none") != "none" for key in FORBIDDEN_MODES):
        raise ValueError(f"{seed}/{arm}: a forbidden bridge/auxiliary mode is enabled")
    if any(options.get(key) not in (None, [], "") for key in FORBIDDEN_CALIBRATION_OPTIONS):
        raise ValueError(f"{seed}/{arm}: a forbidden NOT_ENTITLED calibration option is configured")
    for token, filename in OUTPUT_OPTIONS.items():
        key = token[2:].replace("-", "_")
        if Path(str(options.get(key, ""))).resolve() != (run_dir / filename).resolve():
            raise ValueError(f"{seed}/{arm}: generated output path mismatch for {token}")
    output_parent = Path(str(options["output_json"])).resolve().parent
    swa_output = Path(str(options["stage195_tail3_parameter_swa_output_dir"])).resolve()
    if output_parent != swa_output or swa_output != run_dir:
        raise ValueError(f"{seed}/{arm}: Stage195 output directory does not equal output-json parent")
    if arm == "baseline":
        arm_ok = (
            exact_float_option(options, "compatible_positive_margin_weight", 0.0)
            and "controlled_integrity_sidecar_path" not in options
            and "expected_integrity_sidecar_semantic_sha256" not in options
        )
    else:
        arm_ok = (
            exact_float_option(options, "compatible_positive_margin_weight", 0.05)
            and Path(str(options.get("controlled_integrity_sidecar_path", ""))).resolve() == sidecar
            and options.get("expected_integrity_sidecar_semantic_sha256") == SIDECAR_SHA256
        )
    if not arm_ok:
        raise ValueError(f"{seed}/{arm}: margin/sidecar arm contract mismatch")
    return {
        "required": required,
        "arm_contract_passed": True,
        "external_ood_bridge_auxiliary_absent": True,
        "constrained_selection_absent": True,
        "not_entitled_shift_absent": True,
        "state_capsules_absent": True,
        "logits_source": 'output["logits"]',
    }


def validate_stage193_gate_rows(
    rows: Any, expected_names: tuple[str, ...], context: str
) -> None:
    if type(rows) is not list or [row.get("gate") for row in rows if type(row) is dict] != list(expected_names):
        raise ValueError(f"{context}: exact gate order mismatch")
    for row in rows:
        ensure_exact_keys(row, set(GATE_HEADER), context)
        if row["passed"] is not True or row["blocking_reason"] != "":
            raise ValueError(f"{context}: gate did not pass cleanly")


def validate_stage193_source(stage193: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = {path.name for path in stage193.iterdir()}
    if entries != STAGE193_OUTPUTS or any(not (stage193 / name).is_file() for name in entries):
        raise ValueError("Stage193-A exact six-output closure mismatch")
    report = read_json(stage193 / "stage193a_tail3_fresh_seed_manifest_report.json")
    ensure_exact_keys(report, STAGE193_REPORT_KEYS, "Stage193-A report")
    required = {
        "stage": "Stage193-A", "decision": STAGE193_READY, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True,
        "exact_six_run_diagnostic_execution_authorized": True,
        "training_for_model_advancement_authorized": False,
        "model_advancement_decision": False, "subsequent_training_authorized": False,
        "external_data_used": False, "checkpoint_loaded": False, "model_loaded": False,
        "capsule_loaded": False, "statistical_significance_claim": False,
        "current_diagnostic_git_commit": STAGE193_RUNTIME_COMMIT,
        "trainer_blob_commit": STAGE193_TRAINER_BLOB_COMMIT,
        "trainer_blob_sha256": STAGE193_TRAINER_BLOB_SHA256,
        "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
        "ordered_runs": list(SOURCE_RUNS), "run_manifest_count": 6,
        "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20,
        "expected_prediction_rows_per_export": 720,
        "expected_state_capsules_per_run": 0,
        "canonical_labels": list(LABELS), "logits_source": 'output["logits"]',
        "exception": None,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Stage193-A READY closure mismatch")
    for key in (
        "run_manifest_count", "expected_trajectory_rows_per_run",
        "expected_prediction_exports_per_run", "expected_prediction_rows_per_export",
        "expected_state_capsules_per_run",
    ):
        if not exact_int(report.get(key)):
            raise ValueError(f"Stage193-A report integer contract mismatch for {key}")
    validate_stage193_gate_rows(
        report["source_and_template_gates"], STAGE193_SOURCE_GATE_NAMES,
        "Stage193-A source/template gates",
    )
    validate_stage193_gate_rows(
        report["precommitted_gates"], STAGE193_PRECOMMITTED_GATE_NAMES,
        "Stage193-A precommitted gates",
    )
    source_identity_value = report["source_identity"]
    if type(source_identity_value) is not dict or source_identity_value.get("passed") is not True:
        raise ValueError("Stage193-A source identity did not pass")
    frozen = report["frozen_source_identities"]
    if type(frozen) is not dict or frozen.get("trainer_blob_commit") != STAGE193_TRAINER_BLOB_COMMIT:
        raise ValueError("Stage193-A frozen trainer identity mismatch")

    rows = read_jsonl(stage193 / "stage193a_run_manifest.jsonl")
    if len(rows) != 6 or [row.get("run") for row in rows] != list(SOURCE_RUNS):
        raise ValueError("Stage193-A JSONL exact ordered six-run closure mismatch")
    for run, row in zip(SOURCE_RUNS, rows):
        ensure_exact_keys(row, STAGE193_ROW_KEYS, f"Stage193-A row {run}")
        seed = int(run[4:7])
        arm = run.split("_", 1)[1]
        row_required = {
            "stage": "Stage193-A", "run": run, "training_seed": seed,
            "split_seed": 174, "arm": arm, "canonical_labels": list(LABELS),
            "trainer_blob_commit": STAGE193_TRAINER_BLOB_COMMIT,
            "trainer_blob_sha256": STAGE193_TRAINER_BLOB_SHA256,
            "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0,
            "logits_source": 'output["logits"]', "runnable": True,
            "diagnostic_only": True, "exact_six_run_diagnostic_execution_authorized": True,
            "training_for_model_advancement_authorized": False,
            "model_advancement_decision": False, "subsequent_training_authorized": False,
            "external_data_used": False,
        }
        if any(row.get(key) != value for key, value in row_required.items()):
            raise ValueError(f"{run}: Stage193-A row contract mismatch")
        for key in (
            "training_seed", "split_seed", "expected_trajectory_rows",
            "expected_prediction_exports", "expected_prediction_rows_per_export",
            "expected_state_capsules",
        ):
            if not exact_int(row.get(key)):
                raise ValueError(f"{run}: Stage193-A strict integer mismatch for {key}")
        argv = row["argv"]
        option_map(argv)
        if row["command_argv"] != ["python", row["trainer_source_path"], *argv]:
            raise ValueError(f"{run}: Stage193-A command argv mismatch")
        options = option_map(argv)
        if arm == "baseline":
            arm_contract = {"compatible_positive_margin_weight": 0.0, "sidecar_access": False}
            arm_ok = (
                exact_float_option(options, "compatible_positive_margin_weight", 0.0)
                and exact_float_option(options, "compatible_positive_margin_logit", 0.0)
                and "controlled_integrity_sidecar_path" not in options
                and "expected_integrity_sidecar_semantic_sha256" not in options
            )
        else:
            arm_contract = {
                "compatible_positive_margin_weight": 0.05,
                "compatible_positive_margin_logit": 0.0,
                "sidecar_path": str(
                    (stage193.parents[0] / STAGE185_BASENAME /
                     "stage185a_controlled_train_integrity_sidecar.jsonl").resolve()
                ),
                "sidecar_semantic_sha256": SIDECAR_SHA256,
            }
            arm_ok = (
                exact_float_option(options, "compatible_positive_margin_weight", 0.05)
                and exact_float_option(options, "compatible_positive_margin_logit", 0.0)
                and options.get("expected_integrity_sidecar_semantic_sha256") == SIDECAR_SHA256
            )
        if row["arm_contract"] != arm_contract or not arm_ok:
            raise ValueError(f"{run}: Stage193-A exact arm contract mismatch")

    matrix_rows = read_csv(stage193 / "stage193a_run_command_matrix.csv", SOURCE_MATRIX_HEADER)
    if len(matrix_rows) != 6 or [row["run"] for row in matrix_rows] != list(SOURCE_RUNS):
        raise ValueError("Stage193-A command matrix order/cardinality mismatch")
    for filename, names in (
        ("stage193a_source_and_template_gate.csv", STAGE193_SOURCE_GATE_NAMES),
        ("stage193a_precommitted_gate.csv", STAGE193_PRECOMMITTED_GATE_NAMES),
    ):
        csv_rows = read_csv(stage193 / filename, GATE_HEADER)
        if [row["gate"] for row in csv_rows] != list(names):
            raise ValueError(f"{filename}: exact gate names mismatch")
        if any(row["passed"] != "True" or row["blocking_reason"] != "" for row in csv_rows):
            raise ValueError(f"{filename}: a source gate is not cleanly passed")
    return report, rows


def analyze(
    args: argparse.Namespace,
    source_gates: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    repo, stage193, stage185, run_root, _ = establish_safe_paths(args)
    blockers: list[str] = []

    def gate(
        rows: list[dict[str, Any]], name: str, required: Any, observed: Any,
        passed: bool, reason: str,
    ) -> None:
        rows.append({
            "gate": name, "required": required, "observed": observed,
            "passed": passed, "blocking_reason": "" if passed else reason,
        })
        if not passed:
            blockers.append(f"{name}: {reason}")

    identity = source_identity(repo, args.current_diagnostic_git_commit)
    gate(
        source_gates, "stage195_runtime_and_p0_blob_identity", True, identity,
        identity["passed"], "Stage195 runtime or frozen P0 byte identity mismatch",
    )
    stage193_report, source_rows = validate_stage193_source(stage193)
    gate(
        source_gates, "stage193a_exact_ready_closure", STAGE193_READY,
        stage193_report["decision"], True, "",
    )
    sidecar = (stage185 / "stage185a_controlled_train_integrity_sidecar.jsonl").resolve()
    if not sidecar.is_file():
        raise ValueError("authoritative Stage185 sidecar JSONL is absent")
    semantic_sha, sidecar_rows = semantic_sidecar_sha(sidecar)
    gate(
        source_gates, "stage185_semantic_sha256", SIDECAR_SHA256, semantic_sha,
        semantic_sha == SIDECAR_SHA256, "Stage185 semantic SHA256 mismatch",
    )
    if blockers:
        raise ValueError("frozen Stage195-A source closure failed")

    source_by_arm: dict[str, list[list[str]]] = {arm: [] for arm in ARMS}
    for row in source_rows:
        source_by_arm[row["arm"]].append(row["argv"])
    templates: dict[str, list[str]] = {}
    for arm in ARMS:
        normalized = [normalized_template(argv) for argv in source_by_arm[arm]]
        equivalent = len(normalized) == 3 and all(item == normalized[0] for item in normalized[1:])
        gate(
            source_gates, f"{arm}_stage193a_template_equivalence", True,
            {"equivalent": equivalent, "template": normalized[0] if normalized else None},
            equivalent, "the three within-arm Stage193-A argv templates differ",
        )
        if equivalent:
            templates[arm] = source_by_arm[arm][0]
    cross_arm_templates = {
        arm: normalized_template(templates[arm], normalize_arm=True)
        for arm in ARMS if arm in templates
    }
    cross_arm_equal = (
        set(cross_arm_templates) == set(ARMS)
        and cross_arm_templates["baseline"] == cross_arm_templates["intervention"]
    )
    gate(
        source_gates, "cross_arm_only_margin_and_sidecar_semantics_differ", True,
        cross_arm_templates, cross_arm_equal,
        "baseline/intervention differ outside the frozen margin/sidecar contract",
    )
    if blockers:
        raise ValueError("Stage193-A argv template closure failed")

    manifests: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    trainer = (repo / "scripts/train_controlled_v6b_minimal.py").resolve()
    for run in RUNS:
        match = re.fullmatch(r"seed(180|181|182)_(baseline|intervention)", run)
        if match is None:
            raise RuntimeError("internal frozen Stage195 run identity failure")
        seed = int(match.group(1))
        arm = match.group(2)
        run_dir = (run_root / run).resolve()
        if run_dir.exists():
            raise ValueError(f"planned Stage195-B run directory already exists: {run_dir}")
        argv, audit = rewrite_argv(templates[arm], seed, run_dir)
        envelope = validate_envelope(argv, arm, seed, sidecar, run_dir)
        options = option_map(argv)
        checkpoint_filename = options.get("selected_checkpoint_filename")
        if (
            type(checkpoint_filename) is not str
            or Path(checkpoint_filename).name != checkpoint_filename
            or checkpoint_filename in ("", ".", "..")
        ):
            raise ValueError(f"{run}: selected checkpoint filename is not one exact filename")
        predictions = [
            str((run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve())
            for epoch in range(1, 21)
        ]
        arm_contract = (
            {
                "compatible_positive_margin_weight": 0.0,
                "compatible_positive_margin_logit": 0.0,
                "controlled_integrity_sidecar_path": None,
                "expected_integrity_sidecar_semantic_sha256": None,
            }
            if arm == "baseline"
            else {
                "compatible_positive_margin_weight": 0.05,
                "compatible_positive_margin_logit": 0.0,
                "controlled_integrity_sidecar_path": str(sidecar),
                "expected_integrity_sidecar_semantic_sha256": SIDECAR_SHA256,
            }
        )
        command_argv = ["python", str(trainer), *argv]
        row = {
            "stage": "Stage195-A", "run": run, "training_seed": seed,
            "split_seed": 174, "arm": arm, "canonical_labels": list(LABELS),
            "trainer_source_path": str(trainer), "trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "trainer_blob_sha256": TRAINER_BLOB_SHA256,
            "stage195_runtime_repository_commit": args.current_diagnostic_git_commit,
            "argv": argv, "command_argv": command_argv, "command": shlex.join(command_argv),
            "planned_run_directory": str(run_dir),
            "planned_output_json_path": str((run_dir / "training_report.json").resolve()),
            "planned_selected_checkpoint_path": str((run_dir / checkpoint_filename).resolve()),
            "expected_trajectory_contract_path": str((run_dir / "stage191_trajectory_contract.json").resolve()),
            "expected_trajectory_ledger_path": str((run_dir / "stage191_trajectory_epoch_metrics.jsonl").resolve()),
            "expected_prediction_export_paths": predictions,
            "expected_stage195_swa_predictions_path": str((run_dir / "stage195_tail3_parameter_swa_predictions.jsonl").resolve()),
            "expected_stage195_swa_metrics_path": str((run_dir / "stage195_tail3_parameter_swa_metrics.json").resolve()),
            "expected_stage195_swa_contract_path": str((run_dir / "stage195_tail3_parameter_swa_contract.json").resolve()),
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720,
            "expected_stage195_swa_prediction_rows": 720,
            "expected_state_capsules": 0, "expected_swa_checkpoints": 0,
            "logits_source": 'output["logits"]', "arm_contract": arm_contract,
            "argv_mutation_audit": audit, "frozen_training_envelope": envelope,
            "runnable": True, "diagnostic_only": True,
            "exact_six_run_diagnostic_execution_authorized": True,
            "model_advancement_authorized": False, "production_swa_selected": False,
            "entitlement_correction_implemented": False, "stage195c_decision_made": False,
            "subsequent_training_authorized": False,
            "statistical_significance_claimed": False, "external_data_used": False,
        }
        ensure_exact_keys(row, ROW_KEYS, f"generated row {run}")
        manifests.append(row)
        matrix.append({key: row[key] for key in MATRIX_HEADER})

    integer_contract = {
        "training_seed": None, "split_seed": 174, "expected_trajectory_rows": 20,
        "expected_prediction_exports": 20, "expected_prediction_rows_per_export": 720,
        "expected_stage195_swa_prediction_rows": 720, "expected_state_capsules": 0,
        "expected_swa_checkpoints": 0,
    }
    integer_ok = all(
        exact_int(row.get(key)) and (expected is None or row.get(key) == expected)
        for row in manifests for key, expected in integer_contract.items()
    )
    gate(
        gates, "manifest_strict_non_bool_integer_contract", True, integer_ok,
        integer_ok, "manifest integer identity/cardinality contract mismatch",
    )
    order_ok = [row["run"] for row in manifests] == list(RUNS)
    gate(gates, "exact_stage195b_run_order", list(RUNS), [row["run"] for row in manifests], order_ok, "run order mismatch")
    paths_ok = all(
        Path(row["planned_output_json_path"]).parent == Path(row["planned_run_directory"])
        and len(row["expected_prediction_export_paths"]) == 20
        for row in manifests
    )
    gate(gates, "planned_artifact_path_and_cardinality_closure", True, paths_ok, paths_ok, "planned artifact closure mismatch")
    run_root_safe = not run_root.exists() or not any(run_root.iterdir())
    gate(gates, "stage195b_run_root_empty_or_absent", True, run_root_safe, run_root_safe, "Stage195-B run root is nonempty")
    authorization_ok = all(
        row["exact_six_run_diagnostic_execution_authorized"] is True
        and row["model_advancement_authorized"] is False
        and row["production_swa_selected"] is False
        and row["entitlement_correction_implemented"] is False
        and row["stage195c_decision_made"] is False
        and row["subsequent_training_authorized"] is False
        and row["statistical_significance_claimed"] is False
        for row in manifests
    )
    gate(gates, "diagnostic_only_authorization", True, authorization_ok, authorization_ok, "authorization scope mismatch")
    if blockers:
        raise ValueError("Stage195-A precommitted gate closure failed")

    report = {
        "stage": "Stage195-A", "decision": READY, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True,
        "exact_six_run_diagnostic_execution_authorized": True,
        "model_advancement_authorized": False, "production_swa_selected": False,
        "entitlement_correction_implemented": False, "stage195c_decision_made": False,
        "subsequent_training_authorized": False, "statistical_significance_claimed": False,
        "stage195b_training_performed": False, "model_loaded": False,
        "checkpoint_loaded": False, "external_data_used": False,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT,
        "trainer_blob_sha256": TRAINER_BLOB_SHA256,
        "stage195_runtime_repository_commit": args.current_diagnostic_git_commit,
        "source_identity": identity,
        "frozen_source_identities": {
            "stage193a_directory": str(stage193),
            "stage193_runtime_repository_commit": STAGE193_RUNTIME_COMMIT,
            "stage193_trainer_blob_commit": STAGE193_TRAINER_BLOB_COMMIT,
            "stage193_trainer_blob_sha256": STAGE193_TRAINER_BLOB_SHA256,
            "stage185_sidecar_directory": str(stage185),
            "stage185_sidecar_path": str(sidecar),
            "stage185_sidecar_rows": sidecar_rows,
            "stage185_sidecar_semantic_sha256": SIDECAR_SHA256,
            "stage195p0_specification_sha256": STAGE195P0_SPEC_SHA256,
        },
        "stage195b_run_root": str(run_root), "ordered_runs": list(RUNS),
        "run_manifest_count": 6, "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20,
        "expected_prediction_rows_per_export": 720,
        "expected_stage195_swa_prediction_rows_per_run": 720,
        "expected_state_capsules_per_run": 0, "expected_swa_checkpoints_per_run": 0,
        "canonical_labels": list(LABELS), "logits_source": 'output["logits"]',
        "source_and_template_gates": source_gates, "precommitted_gates": gates,
        "exception": None,
    }
    ensure_exact_keys(report, REPORT_KEYS, "Stage195-A READY report")
    for key in (
        "run_manifest_count", "expected_trajectory_rows_per_run",
        "expected_prediction_exports_per_run", "expected_prediction_rows_per_export",
        "expected_stage195_swa_prediction_rows_per_run", "expected_state_capsules_per_run",
        "expected_swa_checkpoints_per_run",
    ):
        if not exact_int(report[key]):
            raise ValueError(f"Stage195-A report strict integer mismatch for {key}")
    return report, manifests, matrix


def blocked_report(
    args: argparse.Namespace,
    exc: BaseException,
    source_gates: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "stage": "Stage195-A", "decision": BLOCKED, "runnable": False,
        "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "diagnostic_only": True,
        "exact_six_run_diagnostic_execution_authorized": False,
        "model_advancement_authorized": False, "production_swa_selected": False,
        "entitlement_correction_implemented": False, "stage195c_decision_made": False,
        "subsequent_training_authorized": False, "statistical_significance_claimed": False,
        "stage195b_training_performed": False, "model_loaded": False,
        "checkpoint_loaded": False, "external_data_used": False,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT,
        "trainer_blob_sha256": TRAINER_BLOB_SHA256,
        "stage195_runtime_repository_commit": args.current_diagnostic_git_commit,
        "source_identity": None, "frozen_source_identities": None,
        "stage195b_run_root": str(args.stage195b_run_root.resolve()),
        "ordered_runs": list(RUNS), "run_manifest_count": 0,
        "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20,
        "expected_prediction_rows_per_export": 720,
        "expected_stage195_swa_prediction_rows_per_run": 720,
        "expected_state_capsules_per_run": 0, "expected_swa_checkpoints_per_run": 0,
        "canonical_labels": list(LABELS), "logits_source": 'output["logits"]',
        "source_and_template_gates": source_gates, "precommitted_gates": gates,
        "exception": {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }
    ensure_exact_keys(report, REPORT_KEYS, "Stage195-A BLOCKED report")
    return report


def markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Stage195-A tail-three parameter-SWA manifest", "",
        f"Decision: `{report['decision']}`", "",
        f"- Runnable: {str(report['runnable']).lower()}",
        "- Diagnostic only: true",
        f"- Trainer blob commit: `{report['trainer_blob_commit']}`",
        f"- Trainer blob SHA256: `{report['trainer_blob_sha256']}`",
        f"- Stage195 runtime repository commit: `{report['stage195_runtime_repository_commit']}`",
        f"- Exact six-run diagnostic execution authorized: {str(report['exact_six_run_diagnostic_execution_authorized']).lower()}",
        "- Model advancement authorized: false",
        "- Production SWA selected: false",
        "- Entitlement correction implemented: false",
        "- Stage195-C decision made: false",
        "- Subsequent training authorized: false",
        "- Statistical-significance claimed: false", "",
        "READY authorizes only the six exact Stage195-B diagnostic runs recorded in the manifest.",
        "It does not authorize model advancement, production selection, entitlement correction, Stage195-C, or later training.",
        "",
    ])


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return value


def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError("generated CSV row exact-key closure mismatch")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return stream.getvalue()


def render_outputs(
    report: dict[str, Any], manifests: list[dict[str, Any]], matrix: list[dict[str, Any]],
    source_gates: list[dict[str, Any]], gates: list[dict[str, Any]],
) -> dict[str, str]:
    ensure_exact_keys(report, REPORT_KEYS, "rendered Stage195-A report")
    ready = report["decision"] == READY
    if ready:
        if len(manifests) != 6 or [row["run"] for row in manifests] != list(RUNS):
            raise ValueError("READY JSONL exact six-row order closure mismatch")
        if len(matrix) != 6:
            raise ValueError("READY command matrix cardinality mismatch")
    elif manifests or matrix:
        raise ValueError("BLOCKED outputs must not expose manifest or command rows")
    for row in manifests:
        ensure_exact_keys(row, ROW_KEYS, "rendered Stage195-A JSONL row")
    for row in source_gates + gates:
        ensure_exact_keys(row, set(GATE_HEADER), "rendered Stage195-A gate row")
    jsonl_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in manifests
    )
    contents = {
        OUTPUTS["json"]: json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        OUTPUTS["md"]: markdown(report),
        OUTPUTS["jsonl"]: jsonl_text,
        OUTPUTS["matrix"]: render_csv(MATRIX_HEADER, matrix),
        OUTPUTS["source"]: render_csv(GATE_HEADER, source_gates),
        OUTPUTS["gates"]: render_csv(GATE_HEADER, gates),
    }
    if set(contents) != set(OUTPUTS.values()):
        raise RuntimeError("Stage195-A exact six-output name closure mismatch")
    return contents


def publish_outputs(output: Path, contents: dict[str, str]) -> None:
    if set(contents) != set(OUTPUTS.values()):
        raise ValueError("publication output-name closure mismatch")
    temporary = {name: output / f".{name}.stage195a.tmp" for name in contents}
    targets = {name: output / name for name in contents}
    if any(path.exists() for path in [*temporary.values(), *targets.values()]):
        raise FileExistsError("Stage195-A refuses to overwrite an output or private temporary file")
    try:
        for name, text in contents.items():
            with temporary[name].open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        report_name = OUTPUTS["json"]
        for name in [item for item in contents if item != report_name] + [report_name]:
            os.replace(temporary[name], targets[name])
    except BaseException:
        for path in [*temporary.values(), *targets.values()]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def add_failure_gate(gates: list[dict[str, Any]], exc: BaseException) -> None:
    gates.append({
        "gate": "fail_closed_exception", "required": "no exception",
        "observed": {"type": type(exc).__name__, "message": str(exc)},
        "passed": False, "blocking_reason": f"{type(exc).__name__}: {exc}",
    })


def main() -> int:
    args = parse_args()
    try:
        *_, output = establish_safe_paths(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 2
    output.mkdir(parents=False, exist_ok=True)
    source_gates: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    try:
        report, manifests, matrix = analyze(args, source_gates, gates)
        contents = render_outputs(report, manifests, matrix, source_gates, gates)
        publish_outputs(output, contents)
        return 0
    except BaseException as exc:
        add_failure_gate(gates, exc)
        report = blocked_report(args, exc, source_gates, gates)
        try:
            contents = render_outputs(report, [], [], source_gates, gates)
            publish_outputs(output, contents)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            return 3
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
