#!/usr/bin/env python3
"""Create the frozen, fail-closed Stage196-B1-A six-run manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

STAGE = "Stage196-B1-A"
READY = "STAGE196B1_FRAMEGATE_GRADIENT_OWNERSHIP_MANIFEST_READY"
BLOCKED = "STAGE196B1_FRAMEGATE_GRADIENT_OWNERSHIP_MANIFEST_BLOCKED"
FRAMEGATE_IMPLEMENTATION_GIT_COMMIT = "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"
TRAINER_RELATIVE = Path("scripts/train_controlled_v6b_minimal.py")
DATA_RELATIVE = Path("data/controlled_v5_v3_without_time_swap.jsonl")
DATA_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
RUN_ROOT_NAME = "stage196b1_framegate_gradient_ownership_runs"
SUPERSEDED_MANIFEST_BASENAME = (
    "stage196b1_framegate_gradient_ownership_manifest_20260719_174334"
)
SEEDS = (183, 184, 185)
SPLIT_SEED = 174
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)

OUTPUT_NAMES = (
    "stage196b1_run_manifest.json",
    "stage196b1_run_manifest.csv",
    "stage196b1_run_commands.jsonl",
    "stage196b1_manifest_report.json",
    "stage196b1_manifest_report.md",
    "stage196b1_source_closure.csv",
    "stage196b1_precommitted_contract.csv",
)
REPORT_NAME = "stage196b1_manifest_report.json"

MANIFEST_CSV_HEADER = [
    "run", "seed", "split_seed", "arm", "frame_downstream_gradient_mode",
    "run_directory", "training_report_path", "clean_dev_predictions_path",
    "clean_dev_scalars_path", "trajectory_contract_path",
    "trajectory_ledger_path", "trajectory_prediction_paths", "stdout_path",
    "stderr_path", "argv", "command_argv", "forbidden_feature_assertions",
    "expected_runtime_provenance",
]
SOURCE_HEADER = ["source", "path", "expected", "observed", "passed", "blocking_reason"]
CONTRACT_HEADER = ["gate", "required", "observed", "passed", "blocking_reason"]

FORBIDDEN_FEATURE_ASSERTIONS = {
    "time_swap_used_in_main_training": False,
    "external_training_data_used": False,
    "external_eval_enabled": False,
    "external_metrics_used_for_selection": False,
    "stage195_parameter_swa_enabled": False,
    "compatible_positive_margin_enabled": False,
    "new_loss_enabled": False,
    "threshold_tuning_enabled": False,
    "calibration_enabled": False,
}

FORBIDDEN_OPTION_FRAGMENTS = (
    "stage195", "stage193-tail3", "integrity-sidecar", "external", "ood-",
    "bridge", "selector", "composer", "calibrat", "threshold", "contrastive",
    "stage174c", "stage175b", "stage177c", "support-anchor", "pairwise",
    "boundary-loss", "frame-violation-loss", "temporal-aux", "temporal-diagnostic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_output(repo: Path, arguments: list[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=False, capture_output=True, shell=False
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", errors="strict").strip()


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError("CSV row exact-key closure mismatch")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return handle.getvalue()


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def gate(
    rows: list[dict[str, Any]], gate_name: str, required: Any, observed: Any, passed: bool,
    reason: str,
) -> None:
    rows.append({
        "gate": gate_name,
        "required": required,
        "observed": observed,
        "passed": passed,
        "blocking_reason": "" if passed else reason,
    })


def source_gate(
    rows: list[dict[str, Any]], source: str, path: Path | str, expected: Any,
    observed: Any, passed: bool, reason: str,
) -> None:
    rows.append({
        "source": source,
        "path": str(path),
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "blocking_reason": "" if passed else reason,
    })


def option_map(argv: list[str]) -> dict[str, str | bool]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a list of strings")
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            raise ValueError(f"unsupported argv token {token!r}")
        if token in options:
            raise ValueError(f"duplicate argv option {token!r}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            options[token] = argv[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return options


def normalized_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    replace = {
        "--seed": "<SEED>",
        "--frame-downstream-gradient-mode": "<MODE>",
        "--stage115-clean-dev-scalar-output-jsonl": "<SCALARS_PATH>",
        "--output-json": "<REPORT_PATH>",
        "--output-predictions-json": "<PREDICTIONS_PATH>",
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        normalized.append(token)
        if token in replace:
            normalized.append(replace[token])
            index += 2
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            normalized.append(argv[index + 1])
            index += 2
        else:
            index += 1
    return normalized


def expected_outputs(run_dir: Path) -> dict[str, Any]:
    trajectory_predictions = [
        str(run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl")
        for epoch in range(1, 21)
    ]
    return {
        "run_directory": str(run_dir),
        "training_report": str(run_dir / "training_report.json"),
        "clean_dev_predictions": str(run_dir / "clean_dev_predictions.json"),
        "clean_dev_scalars": str(run_dir / "clean_dev_scalars.jsonl"),
        "selected_checkpoint": str(run_dir / "selected_checkpoint.pt"),
        "trajectory_contract": str(run_dir / "stage191_trajectory_contract.json"),
        "trajectory_ledger": str(run_dir / "stage191_trajectory_epoch_metrics.jsonl"),
        "trajectory_predictions": trajectory_predictions,
        "required_tail_epochs": [18, 19, 20],
        "stdout": str(run_dir / "stdout.log"),
        "stderr": str(run_dir / "stderr.log"),
        "expected_trajectory_rows": 20,
        "expected_prediction_exports": 20,
        "expected_prediction_rows_per_export": 720,
        "expected_state_capsules": 0,
        "expected_stage195_swa_artifacts": 0,
    }


def build_argv(seed: int, mode: str, outputs: dict[str, Any]) -> list[str]:
    return [
        "--data", DATA_RELATIVE.as_posix(),
        "--backbone", "mamba",
        "--model-name", "state-spaces/mamba-130m-hf",
        "--architecture", "v6b_minimal",
        "--device", "cuda",
        "--epochs", "20",
        "--seed", str(seed),
        "--split-seed", str(SPLIT_SEED),
        "--frame-downstream-gradient-mode", mode,
        "--stage196b1-framegate-gradient-ownership-observability",
        "--stage115-clean-dev-scalar-output-jsonl", outputs["clean_dev_scalars"],
        "--output-json", outputs["training_report"],
        "--output-predictions-json", outputs["clean_dev_predictions"],
        "--compatible-positive-margin-weight", "0.0",
        "--compatible-positive-margin-logit", "0.0",
        "--lr", "0.001",
        "--freeze-encoder", "true",
        "--freeze-a-log", "true",
        "--max-length", "128",
        "--dev-ratio", "0.2",
        "--gradient-accumulation-steps", "1",
        "--class-weighting", "none",
        "--select-metric", "final_macro_f1",
        "--flag-source", "controlled_heuristic",
        "--save-selected-checkpoint",
        "--selected-checkpoint-filename", "selected_checkpoint.pt",
    ]


def runtime_provenance(seed: int, mode: str) -> dict[str, Any]:
    return {
        "training_seed": seed,
        "configured_split_seed": SPLIT_SEED,
        "resolved_split_seed": SPLIT_SEED,
        "split_seed_explicit": True,
        "split_policy": "fixed_explicit_split_seed",
        "trajectory_observability_mode": "stage196b1_framegate_gradient_ownership",
        "stage191_trajectory_observability_implementation_reused": True,
        "stage191_frozen_seed_contract_modified": False,
        "stage196b1_authorized_training_seeds": list(SEEDS),
        "observability_mode": "stage196b1_framegate_gradient_ownership",
        "authorized_training_seeds": list(SEEDS),
        "stage196b1_framegate_gradient_ownership_observability": True,
        "stage191_trajectory_replay_observability": False,
        "stage193_tail3_fresh_seed_observability": False,
        "stage195_tail3_parameter_swa_causal_test": False,
        "device": "cuda",
        "backbone": "mamba",
        "model_name": "state-spaces/mamba-130m-hf",
        "architecture": "v6b_minimal",
        "frame_downstream_gradient_mode": mode,
        "frame_direct_loss_active": True,
        "frame_direct_loss_weight": 1.0,
        "frame_downstream_forward_value_changed": False,
        "framegate_nonframe_output_gradient_blocked": mode == "frame_local_only",
        "freeze_encoder": True,
        "freeze_a_log": True,
        "shared_encoder_trainable": False,
        "shared_encoder_gradient_fully_isolated": True,
        "shared_encoder_isolation_source": "frozen_runtime_configuration",
        "framegate_gradient_ownership_intervention_changed_encoder_freeze_state": False,
    }


def validate_run_argv(row: dict[str, Any]) -> None:
    argv = row["argv"]
    options = option_map(argv)
    expected = {
        "--data": DATA_RELATIVE.as_posix(),
        "--backbone": "mamba",
        "--model-name": "state-spaces/mamba-130m-hf",
        "--architecture": "v6b_minimal",
        "--device": "cuda",
        "--epochs": "20",
        "--seed": str(row["seed"]),
        "--split-seed": str(SPLIT_SEED),
        "--frame-downstream-gradient-mode": row["frame_downstream_gradient_mode"],
        "--stage196b1-framegate-gradient-ownership-observability": True,
        "--stage115-clean-dev-scalar-output-jsonl": row["outputs"]["clean_dev_scalars"],
        "--output-json": row["outputs"]["training_report"],
        "--output-predictions-json": row["outputs"]["clean_dev_predictions"],
        "--compatible-positive-margin-weight": "0.0",
        "--compatible-positive-margin-logit": "0.0",
        "--lr": "0.001",
        "--freeze-encoder": "true",
        "--freeze-a-log": "true",
        "--max-length": "128",
        "--dev-ratio": "0.2",
        "--gradient-accumulation-steps": "1",
        "--class-weighting": "none",
        "--select-metric": "final_macro_f1",
        "--flag-source": "controlled_heuristic",
        "--save-selected-checkpoint": True,
        "--selected-checkpoint-filename": "selected_checkpoint.pt",
    }
    if options != expected:
        raise ValueError(f"{row['run']}: exact frozen argv option closure mismatch")
    lowered = [token.lower() for token in options]
    forbidden = [
        token for token in lowered
        if any(fragment in token for fragment in FORBIDDEN_OPTION_FRAGMENTS)
    ]
    if forbidden:
        raise ValueError(f"{row['run']}: forbidden options present: {forbidden!r}")
    if "time_swap" not in str(options["--data"]) or "without_time_swap" not in str(options["--data"]):
        raise ValueError(f"{row['run']}: main data is not the frozen time-swap-excluded file")


def build_ready(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    trainer = (repo / TRAINER_RELATIVE).resolve()
    data = (repo / DATA_RELATIVE).resolve()
    run_root = (output_dir.parent / RUN_ROOT_NAME).resolve()
    if not repo.is_dir():
        raise ValueError("repo root is not an existing directory")
    if output_dir == repo or run_root == repo or output_dir == run_root:
        raise ValueError("unsafe output or run-root path")
    if not trainer.is_file() or not data.is_file():
        raise FileNotFoundError("trainer and data files must both exist")
    if not re.fullmatch(r"[0-9a-f]{40}", args.current_git_commit):
        raise ValueError("current git commit must be a lowercase full 40-character SHA")

    source_rows: list[dict[str, Any]] = []
    head = str(git_output(repo, ["rev-parse", "HEAD"]))
    source_gate(source_rows, "current_git_commit", repo, args.current_git_commit, head,
                head == args.current_git_commit, "current git commit differs from HEAD")
    trainer_bytes = trainer.read_bytes()
    committed_trainer_bytes = git_output(
        repo,
        ["show", f"{args.current_git_commit}:{TRAINER_RELATIVE.as_posix()}"],
        binary=True,
    )
    assert isinstance(committed_trainer_bytes, bytes)
    git_output(
        repo,
        [
            "merge-base", "--is-ancestor",
            FRAMEGATE_IMPLEMENTATION_GIT_COMMIT,
            args.current_git_commit,
        ],
    )
    current_trainer_sha = bytes_sha256(trainer_bytes)
    committed_trainer_sha = bytes_sha256(committed_trainer_bytes)
    source_gate(source_rows, "trainer_exists", trainer, True, trainer.is_file(), trainer.is_file(),
                "trainer file is absent")
    source_gate(source_rows, "trainer_git_commit", TRAINER_RELATIVE,
                args.current_git_commit, args.current_git_commit, True,
                "runtime trainer commit mismatch")
    source_gate(source_rows, "framegate_implementation_git_commit", TRAINER_RELATIVE,
                FRAMEGATE_IMPLEMENTATION_GIT_COMMIT,
                FRAMEGATE_IMPLEMENTATION_GIT_COMMIT, True,
                "FrameGate implementation commit is not an ancestor")
    source_gate(source_rows, "trainer_byte_identity", trainer, committed_trainer_sha,
                current_trainer_sha, current_trainer_sha == committed_trainer_sha,
                "current trainer differs from the supplied runtime commit")
    trainer_text = trainer_bytes.decode("utf-8", errors="strict")
    markers = (
        "--frame-downstream-gradient-mode", "frame_local_only",
        "framegate_nonframe_output_gradient_blocked", "frame_direct_loss_weight",
        "freeze_encoder", "freeze_a_log", "shared_encoder_trainable",
        "shared_encoder_gradient_fully_isolated",
        "shared_encoder_isolation_source",
        "framegate_gradient_ownership_intervention_changed_encoder_freeze_state",
        "--stage196b1-framegate-gradient-ownership-observability",
        "STAGE196B1_FRAMEGATE_GRADIENT_OWNERSHIP_SEEDS = (183, 184, 185)",
        "stage196b1_framegate_gradient_ownership",
    )
    missing_markers = [marker for marker in markers if marker not in trainer_text]
    source_gate(source_rows, "trainer_runtime_contract_markers", trainer, list(markers),
                missing_markers, not missing_markers,
                "trainer is missing required FrameGate ownership provenance markers")
    observed_data_sha = file_sha256(data)
    source_gate(source_rows, "data_exists", data, True, data.is_file(), data.is_file(),
                "data file is absent")
    source_gate(source_rows, "data_sha256", data, DATA_SHA256, observed_data_sha,
                observed_data_sha == DATA_SHA256, "data SHA-256 differs from frozen identity")
    if any(row["passed"] is not True for row in source_rows):
        reasons = [row["blocking_reason"] for row in source_rows if row["blocking_reason"]]
        raise ValueError("; ".join(reasons))

    run_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    all_artifact_paths: list[str] = []
    for seed in SEEDS:
        for mode in MODES:
            run = f"seed{seed}_{mode}"
            arm = "baseline" if mode == "joint" else "intervention"
            run_dir = run_root / run
            outputs = expected_outputs(run_dir)
            argv = build_argv(seed, mode, outputs)
            command_argv = [sys.executable, str(trainer), *argv]
            provenance = runtime_provenance(seed, mode)
            row = {
                "run": run,
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "arm": arm,
                "frame_downstream_gradient_mode": mode,
                "run_directory": str(run_dir),
                "argv": argv,
                "command_argv": command_argv,
                "outputs": outputs,
                "forbidden_feature_assertions": dict(FORBIDDEN_FEATURE_ASSERTIONS),
                "expected_runtime_provenance": provenance,
                "execution_not_performed": True,
            }
            validate_run_argv(row)
            run_rows.append(row)
            command_rows.append({
                "run": run,
                "arm": arm,
                "frame_downstream_gradient_mode": mode,
                "cwd": str(repo),
                "command_argv": command_argv,
                "stdout_path": outputs["stdout"],
                "stderr_path": outputs["stderr"],
                "shell": False,
                "stop_on_nonzero_return": True,
                "preserve_partial_run_directory": True,
                "execution_not_performed": True,
            })
            for key, value in outputs.items():
                if key == "run_directory" or key.startswith("expected_") or key == "required_tail_epochs":
                    continue
                if isinstance(value, list):
                    all_artifact_paths.extend(value)
                else:
                    all_artifact_paths.append(value)

    contract_rows: list[dict[str, Any]] = []
    gate(contract_rows, "exact_six_run_order", list(RUNS),
         [row["run"] for row in run_rows], [row["run"] for row in run_rows] == list(RUNS),
         "exact run order mismatch")
    gate(contract_rows, "exact_seeds", list(SEEDS), sorted({row["seed"] for row in run_rows}),
         sorted({row["seed"] for row in run_rows}) == list(SEEDS), "seed set mismatch")
    counts = {seed: sum(row["seed"] == seed for row in run_rows) for seed in SEEDS}
    gate(contract_rows, "exactly_two_runs_per_seed", {str(seed): 2 for seed in SEEDS}, counts,
         all(value == 2 for value in counts.values()), "run count per seed mismatch")
    modes_by_seed = {
        seed: [row["frame_downstream_gradient_mode"] for row in run_rows if row["seed"] == seed]
        for seed in SEEDS
    }
    gate(contract_rows, "one_joint_one_frame_local_only_per_seed", list(MODES), modes_by_seed,
         all(value == list(MODES) for value in modes_by_seed.values()), "mode pairing mismatch")
    gate(contract_rows, "split_seed_exact", SPLIT_SEED,
         sorted({row["split_seed"] for row in run_rows}),
         {row["split_seed"] for row in run_rows} == {SPLIT_SEED}, "split seed mismatch")
    gate(contract_rows, "exact_data_path", DATA_RELATIVE.as_posix(),
         sorted({option_map(row["argv"])["--data"] for row in run_rows}),
         all(option_map(row["argv"])["--data"] == DATA_RELATIVE.as_posix() for row in run_rows),
         "data path mismatch")
    gate(contract_rows, "time_swap_excluded_from_main_data", "without_time_swap dataset",
         DATA_RELATIVE.as_posix(), DATA_RELATIVE.name == "controlled_v5_v3_without_time_swap.jsonl",
         "time-swap-containing main data selected")
    identities = [
        (option_map(row["argv"])["--backbone"], option_map(row["argv"])["--device"],
         option_map(row["argv"])["--architecture"]) for row in run_rows
    ]
    gate(contract_rows, "mamba_cuda_v6b_identity", ["mamba", "cuda", "v6b_minimal"], identities,
         all(value == ("mamba", "cuda", "v6b_minimal") for value in identities),
         "model runtime identity mismatch")
    gate(contract_rows, "epochs_exact", 20,
         [option_map(row["argv"])["--epochs"] for row in run_rows],
         all(option_map(row["argv"])["--epochs"] == "20" for row in run_rows),
         "epoch count mismatch")
    freeze_encoder_values = [option_map(row["argv"])["--freeze-encoder"] for row in run_rows]
    freeze_a_log_values = [option_map(row["argv"])["--freeze-a-log"] for row in run_rows]
    freeze_encoder_exact = all(value == "true" for value in freeze_encoder_values)
    freeze_a_log_exact = all(value == "true" for value in freeze_a_log_values)
    gate(contract_rows, "freeze_encoder_exactly_true", ["true"] * 6,
         freeze_encoder_values, freeze_encoder_exact,
         "Stage196-B1 requires --freeze-encoder true in every run")
    gate(contract_rows, "freeze_a_log_exactly_true", ["true"] * 6,
         freeze_a_log_values, freeze_a_log_exact,
         "Stage196-B1 requires --freeze-a-log true in every run")
    freeze_pairwise_equal = all(
        freeze_encoder_values[index] == freeze_encoder_values[index + 1]
        and freeze_a_log_values[index] == freeze_a_log_values[index + 1]
        for index in (0, 2, 4)
    )
    gate(contract_rows, "paired_freeze_arguments_equal", True,
         freeze_pairwise_equal, freeze_pairwise_equal,
         "freeze arguments differ within a seed pair")
    gate(contract_rows, "stage196b1_observability_enabled", True,
         [option_map(row["argv"]).get("--stage196b1-framegate-gradient-ownership-observability") for row in run_rows],
         all(option_map(row["argv"]).get("--stage196b1-framegate-gradient-ownership-observability") is True for row in run_rows),
         "Stage196-B1 observability is not enabled")
    raw_stage191_absent = all(
        "--stage191-trajectory-replay-observability" not in option_map(row["argv"])
        for row in run_rows
    )
    gate(contract_rows, "raw_stage191_mode_absent", True, raw_stage191_absent,
         raw_stage191_absent, "raw Stage191 observability mode is present")
    gate(contract_rows, "stage115_scalar_export_present", True,
         [row["outputs"]["clean_dev_scalars"] for row in run_rows],
         all(option_map(row["argv"])["--stage115-clean-dev-scalar-output-jsonl"] == row["outputs"]["clean_dev_scalars"] for row in run_rows),
         "Stage115 scalar export path mismatch")
    margins_off = all(
        option_map(row["argv"])["--compatible-positive-margin-weight"] == "0.0"
        and option_map(row["argv"])["--compatible-positive-margin-logit"] == "0.0"
        for row in run_rows
    )
    gate(contract_rows, "compatible_positive_margin_exactly_off", [0.0, 0.0], margins_off,
         margins_off, "compatible-positive margin is enabled")
    all_options = [token for row in run_rows for token in option_map(row["argv"])]
    stage195_absent = not any("stage195" in token.lower() for token in all_options)
    gate(contract_rows, "stage195_swa_absent", True, stage195_absent, stage195_absent,
         "Stage195 SWA option present")
    external_bridge_absent = not any(
        any(fragment in token.lower() for fragment in ("external", "ood-", "bridge", "sidecar"))
        for token in all_options
    )
    gate(contract_rows, "external_and_bridge_options_absent", True, external_bridge_absent,
         external_bridge_absent, "external, OOD, bridge, or sidecar option present")
    unique_paths = len(all_artifact_paths) == len(set(all_artifact_paths))
    gate(contract_rows, "all_output_paths_unique", len(all_artifact_paths), len(set(all_artifact_paths)),
         unique_paths, "planned output paths are not unique")
    arrays_only = all(
        type(row["argv"]) is list and type(row["command_argv"]) is list
        and all(type(token) is str for token in row["command_argv"])
        for row in run_rows
    )
    gate(contract_rows, "argv_arrays_no_shell_strings", True, arrays_only, arrays_only,
         "a command is not represented only by argv arrays")
    commit_valid = re.fullmatch(r"[0-9a-f]{40}", args.current_git_commit) is not None
    gate(contract_rows, "current_commit_full_sha", True, args.current_git_commit, commit_valid,
         "current commit is not a full SHA")
    gate(contract_rows, "trainer_and_data_exist", True, [trainer.is_file(), data.is_file()],
         trainer.is_file() and data.is_file(), "trainer or data file is absent")
    normalized = [normalized_argv(row["argv"]) for row in run_rows]
    pairwise_equal = all(value == normalized[0] for value in normalized)
    gate(contract_rows, "paired_arguments_equal_after_allowed_normalization", True,
         pairwise_equal, pairwise_equal, "run argv differs beyond seed, mode, or output paths")
    mode_specific = all(
        row["expected_runtime_provenance"]["framegate_nonframe_output_gradient_blocked"]
        is (row["frame_downstream_gradient_mode"] == "frame_local_only")
        for row in run_rows
    )
    gate(contract_rows, "mode_specific_runtime_ownership_assertions", True, mode_specific,
         mode_specific, "runtime ownership assertion is not mode-specific")
    encoder_provenance_exact = all(
        row["expected_runtime_provenance"]["freeze_encoder"] is True
        and row["expected_runtime_provenance"]["freeze_a_log"] is True
        and row["expected_runtime_provenance"]["shared_encoder_trainable"] is False
        and row["expected_runtime_provenance"]["shared_encoder_gradient_fully_isolated"] is True
        and row["expected_runtime_provenance"]["shared_encoder_isolation_source"]
            == "frozen_runtime_configuration"
        and row["expected_runtime_provenance"][
            "framegate_gradient_ownership_intervention_changed_encoder_freeze_state"
        ] is False
        for row in run_rows
    )
    gate(contract_rows, "frozen_encoder_runtime_provenance_exact", True,
         encoder_provenance_exact, encoder_provenance_exact,
         "expected shared-encoder provenance contradicts the frozen argv")
    paired_ownership_provenance_exact = all(
        {
            key: value for key, value in run_rows[index]["expected_runtime_provenance"].items()
            if key not in ("frame_downstream_gradient_mode", "framegate_nonframe_output_gradient_blocked")
        } == {
            key: value for key, value in run_rows[index + 1]["expected_runtime_provenance"].items()
            if key not in ("frame_downstream_gradient_mode", "framegate_nonframe_output_gradient_blocked")
        }
        and run_rows[index]["expected_runtime_provenance"]["framegate_nonframe_output_gradient_blocked"] is False
        and run_rows[index + 1]["expected_runtime_provenance"]["framegate_nonframe_output_gradient_blocked"] is True
        for index in (0, 2, 4)
    )
    gate(contract_rows, "paired_ownership_provenance_differs_only_by_mode_and_block", True,
         paired_ownership_provenance_exact, paired_ownership_provenance_exact,
         "paired ownership provenance differs beyond mode-specific output blocking")
    output_closure = set(OUTPUT_NAMES) == {
        "stage196b1_run_manifest.json", "stage196b1_run_manifest.csv",
        "stage196b1_run_commands.jsonl", "stage196b1_manifest_report.json",
        "stage196b1_manifest_report.md", "stage196b1_source_closure.csv",
        "stage196b1_precommitted_contract.csv",
    }
    gate(contract_rows, "exact_seven_output_name_closure", list(OUTPUT_NAMES), list(OUTPUT_NAMES),
         output_closure, "output name closure mismatch")
    if any(row["passed"] is not True for row in contract_rows):
        reasons = [row["blocking_reason"] for row in contract_rows if row["blocking_reason"]]
        raise ValueError("; ".join(reasons))

    frozen_common = {
        "data": DATA_RELATIVE.as_posix(),
        "data_sha256": observed_data_sha,
        "backbone": "mamba",
        "model_name": "state-spaces/mamba-130m-hf",
        "architecture": "v6b_minimal",
        "device": "cuda",
        "epochs": 20,
        "split_seed": SPLIT_SEED,
        "select_metric": "final_macro_f1",
        "flag_source": "controlled_heuristic",
        "learning_rate": 0.001,
        "head_learning_rate": None,
        "encoder_learning_rate": None,
        "freeze_encoder": True,
        "freeze_a_log": True,
        "shared_encoder_trainable": False,
        "shared_encoder_gradient_fully_isolated": True,
        "shared_encoder_isolation_source": "frozen_runtime_configuration",
        "framegate_gradient_ownership_intervention_changed_encoder_freeze_state": False,
        "max_length": 128,
        "dev_ratio": 0.2,
        "optimizer": "v5.build_optimizer (AdamW)",
        "weight_decay": None,
        "scheduler": None,
        "train_batch_size": None,
        "eval_batch_size": None,
        "full_split_train_and_eval_forwards": True,
        "gradient_accumulation_steps": 1,
        "fp16": False,
        "class_weighting": "none",
        "save_selected_checkpoint": True,
        "selected_checkpoint_filename": "selected_checkpoint.pt",
        "compatible_positive_margin_weight": 0.0,
        "compatible_positive_margin_logit": 0.0,
        "trajectory_observability_mode": "stage196b1_framegate_gradient_ownership",
        "stage191_trajectory_observability_implementation_reused": True,
        "stage191_frozen_seed_contract_modified": False,
        "stage196b1_authorized_training_seeds": list(SEEDS),
        "stage196b1_framegate_gradient_ownership_observability": True,
        "stage115_clean_dev_scalar_export": True,
    }
    arm_definitions = {
        "baseline": {"frame_downstream_gradient_mode": "joint"},
        "intervention": {"frame_downstream_gradient_mode": "frame_local_only"},
    }
    expected_schema = {
        "runtime_provenance_required_across_training_report_and_trajectory_contract": runtime_provenance(SEEDS[0], "joint"),
        "clean_dev_predictions": "normal trainer clean-dev prediction schema",
        "clean_dev_scalars": "Stage115 JSONL scalar schema",
        "trajectory_contract": "Stage196-B1 mode reusing Stage191 filenames and row schema",
        "trajectory_ledger_rows": 20,
        "trajectory_prediction_exports": 20,
        "trajectory_prediction_rows_per_export": 720,
        "required_exact_epochs": [18, 19, 20],
    }
    causal_metrics = [
        "persistent stable SUPPORT-negative count",
        "recurrent persistent SUPPORT positions",
        "universal persistent SUPPORT positions",
        "FrameGate failure count within persistent SUPPORT negatives",
        "frame probability on Stage196-A recurrent positions",
        "stable-correct SUPPORT preservation",
        "false-entitlement count",
        "false-NOT_ENTITLED count",
        "polarity error count",
        "SUPPORT recall",
        "macro-F1",
    ]
    interpretation_restrictions = [
        "external/OOD performance", "production readiness",
        "shared Mamba representation interference was tested",
        "end-to-end gradient isolation was tested", "unfrozen-backbone behavior is known",
        "intrinsic representation failure",
        "contrastive-loss necessity", "final architecture superiority",
    ]
    runner_policy = {
        "first_run": "seed183_joint",
        "first_run_purpose": "validate joint execution under the Stage196-B1 observability mode",
        "first_frame_local_only_run": "seed183_frame_local_only",
        "first_actual_hook_runtime_test": True,
        "capture_stdout_and_stderr": True,
        "preserve_failed_run_directory": True,
        "stop_immediately_on_nonzero_return": True,
        "continue_after_hook_or_runtime_failure": False,
        "delete_partial_artifacts": False,
        "runtime_failure_is_scientific_evidence": False,
    }
    manifest = {
        "stage": STAGE,
        "decision": READY,
        "runnable": True,
        "blocking_reasons": [],
        "current_git_commit": args.current_git_commit,
        "trainer_path": str(trainer),
        "trainer_git_commit": args.current_git_commit,
        "framegate_implementation_git_commit": FRAMEGATE_IMPLEMENTATION_GIT_COMMIT,
        "trainer_sha256": current_trainer_sha,
        "data_path": DATA_RELATIVE.as_posix(),
        "data_sha256": observed_data_sha,
        "run_root": str(run_root),
        "ordered_runs": list(RUNS),
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "arm_definitions": arm_definitions,
        "frozen_common_configuration": frozen_common,
        "runs": run_rows,
        "forbidden_feature_assertions": dict(FORBIDDEN_FEATURE_ASSERTIONS),
        "expected_runtime_provenance_assertions": {
            row["run"]: row["expected_runtime_provenance"] for row in run_rows
        },
        "expected_output_schema": expected_schema,
        "future_runner_policy": runner_policy,
        "stage196b1c_precommitted_causal_metrics": causal_metrics,
        "numeric_success_threshold_selected": False,
        "interpretation_restrictions": interpretation_restrictions,
        "authorized_causal_claim_scope": "under a frozen Mamba encoder, direct non-frame gradient paths through FrameGate outputs into FrameGate-owned trainable parameters",
        "superseded_manifest_directory": SUPERSEDED_MANIFEST_BASENAME,
        "superseded_manifest_mechanically_valid": True,
        "superseded_before_execution_for_semantic_provenance_error": True,
        "new_timestamped_manifest_required_after_commit": True,
        "execution_not_performed": True,
    }
    report = {
        "stage": STAGE,
        "decision": READY,
        "runnable": True,
        "blocking_reasons": [],
        "current_git_commit": args.current_git_commit,
        "trainer_git_commit": args.current_git_commit,
        "framegate_implementation_git_commit": FRAMEGATE_IMPLEMENTATION_GIT_COMMIT,
        "data_sha256": observed_data_sha,
        "run_root": str(run_root),
        "ordered_runs": list(RUNS),
        "run_count": len(run_rows),
        "pairwise_arguments_equal": pairwise_equal,
        "output_file_count": len(OUTPUT_NAMES),
        "source_closure_passed": True,
        "precommitted_contract_passed": True,
        "first_frame_local_only_run_is_first_hook_runtime_test": True,
        "stage196b1c_causal_metrics": causal_metrics,
        "numeric_success_threshold_selected": False,
        "training_executed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "external_evaluation_executed": False,
        "execution_not_performed": True,
        "superseded_manifest_directory": SUPERSEDED_MANIFEST_BASENAME,
        "superseded_manifest_mechanically_valid": True,
        "superseded_before_execution_for_semantic_provenance_error": True,
        "new_timestamped_manifest_required_after_commit": True,
        "remaining_runtime_risks": [
            "The first frame_local_only run is the first actual hook-path runtime test.",
            "The dedicated Stage196-B1 mode and per-epoch seed checks have not been runtime-executed.",
            "The corrected frozen-encoder provenance contract has not been runtime-executed.",
            "CUDA, dependency, memory, and full-duration training availability are not tested here.",
        ],
        "exception": None,
    }
    manifest_csv_rows = []
    for row in run_rows:
        outputs = row["outputs"]
        manifest_csv_rows.append({
            "run": row["run"], "seed": row["seed"], "split_seed": row["split_seed"],
            "arm": row["arm"],
            "frame_downstream_gradient_mode": row["frame_downstream_gradient_mode"],
            "run_directory": row["run_directory"],
            "training_report_path": outputs["training_report"],
            "clean_dev_predictions_path": outputs["clean_dev_predictions"],
            "clean_dev_scalars_path": outputs["clean_dev_scalars"],
            "trajectory_contract_path": outputs["trajectory_contract"],
            "trajectory_ledger_path": outputs["trajectory_ledger"],
            "trajectory_prediction_paths": outputs["trajectory_predictions"],
            "stdout_path": outputs["stdout"], "stderr_path": outputs["stderr"],
            "argv": row["argv"], "command_argv": row["command_argv"],
            "forbidden_feature_assertions": row["forbidden_feature_assertions"],
            "expected_runtime_provenance": row["expected_runtime_provenance"],
        })
    contents = {
        "stage196b1_run_manifest.json": json_text(manifest),
        "stage196b1_run_manifest.csv": render_csv(MANIFEST_CSV_HEADER, manifest_csv_rows),
        "stage196b1_run_commands.jsonl": jsonl_text(command_rows),
        "stage196b1_manifest_report.json": json_text(report),
        "stage196b1_manifest_report.md": render_report_markdown(report),
        "stage196b1_source_closure.csv": render_csv(SOURCE_HEADER, source_rows),
        "stage196b1_precommitted_contract.csv": render_csv(CONTRACT_HEADER, contract_rows),
    }
    return contents, report


def render_report_markdown(report: dict[str, Any]) -> str:
    risks = "\n".join(f"- {risk}" for risk in report["remaining_runtime_risks"])
    runs = "\n".join(f"{index}. `{run}`" for index, run in enumerate(report["ordered_runs"], 1))
    return (
        "# Stage196-B1-A FrameGate gradient-ownership manifest report\n\n"
        f"- Decision: `{report['decision']}`\n"
        f"- Runnable: `{str(report['runnable']).lower()}`\n"
        f"- Current Git commit: `{report['current_git_commit']}`\n"
        f"- Runtime trainer commit: `{report['trainer_git_commit']}`\n"
        f"- FrameGate implementation commit: `{report['framegate_implementation_git_commit']}`\n"
        f"- Data SHA-256: `{report['data_sha256']}`\n"
        f"- Pairwise arguments equal after allowed normalization: "
        f"`{str(report['pairwise_arguments_equal']).lower()}`\n"
        "- Training executed: `false`\n"
        "- Numeric success threshold selected: `false`\n\n"
        "## Superseded pre-execution manifest\n\n"
        f"`{report['superseded_manifest_directory']}` is mechanically valid but "
        "superseded before execution because its shared-encoder provenance was "
        "semantically incorrect. It is preserved; a new timestamped manifest is "
        "required after commit.\n\n"
        "## Frozen run order\n\n"
        f"{runs}\n\n"
        "## First-run hook policy\n\n"
        "Run 1 validates joint execution under the Stage196-B1 mode. "
        "Run 2 is the first actual hook-path runtime test. "
        "The future runner captures stdout/stderr, preserves partial artifacts, and "
        "stops on the first nonzero return. Runtime failure is not scientific evidence.\n\n"
        "## Remaining runtime risks\n\n"
        f"{risks}\n"
    )


def blocked_contents(args: argparse.Namespace, exc: BaseException) -> dict[str, str]:
    exception = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    report = {
        "stage": STAGE,
        "decision": BLOCKED,
        "runnable": False,
        "blocking_reasons": [str(exc)],
        "current_git_commit": args.current_git_commit,
        "trainer_git_commit": args.current_git_commit,
        "framegate_implementation_git_commit": FRAMEGATE_IMPLEMENTATION_GIT_COMMIT,
        "data_sha256": None,
        "run_root": str((args.output_dir.resolve().parent / RUN_ROOT_NAME).resolve()),
        "ordered_runs": list(RUNS),
        "run_count": 0,
        "pairwise_arguments_equal": False,
        "output_file_count": len(OUTPUT_NAMES),
        "source_closure_passed": False,
        "precommitted_contract_passed": False,
        "first_frame_local_only_run_is_first_hook_runtime_test": True,
        "stage196b1c_causal_metrics": [],
        "numeric_success_threshold_selected": False,
        "training_executed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "external_evaluation_executed": False,
        "execution_not_performed": True,
        "superseded_manifest_directory": SUPERSEDED_MANIFEST_BASENAME,
        "superseded_manifest_mechanically_valid": True,
        "superseded_before_execution_for_semantic_provenance_error": True,
        "new_timestamped_manifest_required_after_commit": True,
        "remaining_runtime_risks": [str(exc)],
        "exception": exception,
    }
    manifest = {
        "stage": STAGE, "decision": BLOCKED, "runnable": False,
        "blocking_reasons": [str(exc)], "current_git_commit": args.current_git_commit,
        "trainer_path": str((args.repo_root.resolve() / TRAINER_RELATIVE).resolve()),
        "trainer_git_commit": args.current_git_commit,
        "framegate_implementation_git_commit": FRAMEGATE_IMPLEMENTATION_GIT_COMMIT, "data_path": DATA_RELATIVE.as_posix(),
        "data_sha256": None, "run_root": report["run_root"], "ordered_runs": list(RUNS),
        "seeds": list(SEEDS), "split_seed": SPLIT_SEED, "arm_definitions": {},
        "frozen_common_configuration": {}, "runs": [],
        "forbidden_feature_assertions": dict(FORBIDDEN_FEATURE_ASSERTIONS),
        "expected_runtime_provenance_assertions": {}, "expected_output_schema": {},
        "authorized_causal_claim_scope": "under a frozen Mamba encoder, direct non-frame gradient paths through FrameGate outputs into FrameGate-owned trainable parameters",
        "superseded_manifest_directory": SUPERSEDED_MANIFEST_BASENAME,
        "superseded_manifest_mechanically_valid": True,
        "superseded_before_execution_for_semantic_provenance_error": True,
        "new_timestamped_manifest_required_after_commit": True,
        "execution_not_performed": True, "exception": exception,
    }
    source_row = {
        "source": "manifest_generation", "path": str(args.output_dir), "expected": READY,
        "observed": BLOCKED, "passed": False, "blocking_reason": str(exc),
    }
    contract_row = {
        "gate": "manifest_generation", "required": READY, "observed": BLOCKED,
        "passed": False, "blocking_reason": str(exc),
    }
    return {
        "stage196b1_run_manifest.json": json_text(manifest),
        "stage196b1_run_manifest.csv": render_csv(MANIFEST_CSV_HEADER, []),
        "stage196b1_run_commands.jsonl": "",
        "stage196b1_manifest_report.json": json_text(report),
        "stage196b1_manifest_report.md": render_report_markdown(report),
        "stage196b1_source_closure.csv": render_csv(SOURCE_HEADER, [source_row]),
        "stage196b1_precommitted_contract.csv": render_csv(CONTRACT_HEADER, [contract_row]),
    }


def publish(output_dir: Path, contents: dict[str, str]) -> None:
    if set(contents) != set(OUTPUT_NAMES):
        raise RuntimeError("exact seven-output publication closure mismatch")
    temporary: list[Path] = []
    try:
        for name in OUTPUT_NAMES:
            if (output_dir / name).exists():
                raise FileExistsError(f"refusing to overwrite {output_dir / name}")
        for name in OUTPUT_NAMES:
            temporary_path = output_dir / f".{name}.stage196b1.tmp"
            if temporary_path.exists():
                raise FileExistsError(f"private temporary path already exists: {temporary_path}")
            temporary_path.write_text(contents[name], encoding="utf-8", newline="\n")
            temporary.append(temporary_path)
        publication_order = [name for name in OUTPUT_NAMES if name != REPORT_NAME] + [REPORT_NAME]
        for name in publication_order:
            temporary_path = output_dir / f".{name}.stage196b1.tmp"
            os.replace(temporary_path, output_dir / name)
            temporary.remove(temporary_path)
    finally:
        for path in temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        print("Stage196-B1-A refuses a nonempty output directory", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        contents, _ = build_ready(args)
        publish(output_dir, contents)
        return 0
    except BaseException as exc:
        try:
            if any(output_dir.iterdir()):
                raise
            publish(output_dir, blocked_contents(args, exc))
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
