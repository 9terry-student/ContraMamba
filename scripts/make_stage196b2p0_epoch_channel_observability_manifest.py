#!/usr/bin/env python3
"""Create the fail-closed Stage196-B2-P0 six-run observability manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

STAGE = "Stage196-B2-P0"
READY = "STAGE196B2P0_EPOCH_CHANNEL_OBSERVABILITY_MANIFEST_READY"
BLOCKED = "STAGE196B2P0_EPOCH_CHANNEL_OBSERVABILITY_MANIFEST_BLOCKED"
B1_RUNTIME_COMMIT = "9835cbbf86d83aca0964821669e63f7f6deb1c59"
FRAMEGATE_ORIGIN_COMMIT = "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"
DATA = Path("data/controlled_v5_v3_without_time_swap.jsonl")
DATA_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
TRAINER = Path("scripts/train_controlled_v6b_minimal.py")
GENERATOR = Path("scripts/make_stage196b2p0_epoch_channel_observability_manifest.py")
RUN_ROOT = "stage196b2p0_epoch_channel_observability_runs"
SEEDS = (183, 184, 185)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)
FIELDS = (
    "id", "source_row_id", "dev_position", "gold_label", "prediction",
    "intervention_type", "frame_probability", "predicate_coverage_probability",
    "sufficiency_probability", "polarity_support_margin", "entitlement_probability",
    "support_probability", "not_entitled_probability", "support_logit",
    "not_entitled_logit", "epoch", "training_seed", "frame_downstream_gradient_mode",
)
OUTPUTS = (
    "stage196b2p0_manifest.json", "stage196b2p0_run_commands.jsonl",
    "stage196b2p0_expected_outputs.json", "stage196b2p0_pairing_contract.json",
    "stage196b2p0_source_closure.json", "stage196b2p0_execution_order.txt",
    "stage196b2p0_manifest_report.md",
)
FORBIDDEN_FRAGMENTS = (
    "--ood-", "--external", "--stage43", "--stage57", "--stage66", "--stage75",
    "--stage80", "--bridge", "--calibrat", "--threshold", "--stage195",
    "--stage191-save-trajectory-state-capsules", "--controlled-integrity-sidecar",
    "--expected-integrity-sidecar", "--smoke", "--loss-sweep", "--max-train-records",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--stage196b1-runtime-git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Parent for a newly timestamped manifest directory.")
    return parser.parse_args()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def git(repo: Path, argv: list[str], binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *argv], cwd=repo, shell=False,
                            check=False, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()

def option_map(argv: list[str]) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if type(token) is not str or not token.startswith("--") or token in result:
            raise ValueError(f"invalid or duplicate option token: {token!r}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[token] = argv[index + 1]
            index += 2
        else:
            result[token] = True
            index += 1
    return result

def run_outputs(run_dir: Path) -> dict[str, Any]:
    channels = [str(run_dir / f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl")
                for epoch in range(1, 21)]
    return {
        "run_directory": str(run_dir),
        "training_report": str(run_dir / "training_report.json"),
        "clean_dev_predictions": str(run_dir / "clean_dev_predictions.json"),
        "clean_dev_scalars": str(run_dir / "clean_dev_scalars.jsonl"),
        "selected_checkpoint": str(run_dir / "selected_checkpoint.pt"),
        "trajectory_contract": str(run_dir / "stage191_trajectory_contract.json"),
        "trajectory_ledger": str(run_dir / "stage191_trajectory_epoch_metrics.jsonl"),
        "trajectory_predictions": [str(run_dir / f"stage191_dev_predictions_epoch_{e:03d}.jsonl")
                                   for e in range(1, 21)],
        "epoch_channel_sidecars": channels,
        "epoch_channel_file_count": 20,
        "epoch_channel_rows_per_file": 720,
        "epoch_channel_required_fields": list(FIELDS),
        "state_capsule_count": 0,
    }

def build_argv(seed: int, mode: str, outputs: dict[str, Any]) -> list[str]:
    return [
        "--data", DATA.as_posix(), "--backbone", "mamba",
        "--model-name", "state-spaces/mamba-130m-hf", "--architecture", "v6b_minimal",
        "--device", "cuda", "--epochs", "20", "--seed", str(seed),
        "--split-seed", "174", "--frame-downstream-gradient-mode", mode,
        "--stage196b1-framegate-gradient-ownership-observability",
        "--stage196b2p0-epoch-channel-observability",
        "--stage115-clean-dev-scalar-output-jsonl", outputs["clean_dev_scalars"],
        "--output-json", outputs["training_report"],
        "--output-predictions-json", outputs["clean_dev_predictions"],
        "--compatible-positive-margin-weight", "0.0",
        "--compatible-positive-margin-logit", "0.0", "--lr", "0.001",
        "--freeze-encoder", "true", "--freeze-a-log", "true", "--max-length", "128",
        "--dev-ratio", "0.2", "--gradient-accumulation-steps", "1",
        "--class-weighting", "none", "--select-metric", "final_macro_f1",
        "--flag-source", "controlled_heuristic", "--save-selected-checkpoint",
        "--selected-checkpoint-filename", "selected_checkpoint.pt",
    ]

def normalized(argv: list[str]) -> list[str]:
    replace = {
        "--seed": "<SEED>", "--frame-downstream-gradient-mode": "<MODE>",
        "--stage115-clean-dev-scalar-output-jsonl": "<SCALARS>",
        "--output-json": "<REPORT>", "--output-predictions-json": "<PREDICTIONS>",
    }
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        result.append(token)
        if token in replace:
            result.append(replace[token])
            index += 2
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result.append(argv[index + 1])
            index += 2
        else:
            index += 1
    return result

def validate_command(row: dict[str, Any]) -> None:
    options = option_map(row["argv"])
    expected = {
        "--data": DATA.as_posix(), "--backbone": "mamba",
        "--model-name": "state-spaces/mamba-130m-hf", "--architecture": "v6b_minimal",
        "--device": "cuda", "--epochs": "20", "--seed": str(row["seed"]),
        "--split-seed": "174", "--frame-downstream-gradient-mode": row["mode"],
        "--stage196b1-framegate-gradient-ownership-observability": True,
        "--stage196b2p0-epoch-channel-observability": True,
        "--stage115-clean-dev-scalar-output-jsonl": row["outputs"]["clean_dev_scalars"],
        "--output-json": row["outputs"]["training_report"],
        "--output-predictions-json": row["outputs"]["clean_dev_predictions"],
        "--compatible-positive-margin-weight": "0.0",
        "--compatible-positive-margin-logit": "0.0", "--lr": "0.001",
        "--freeze-encoder": "true", "--freeze-a-log": "true", "--max-length": "128",
        "--dev-ratio": "0.2", "--gradient-accumulation-steps": "1",
        "--class-weighting": "none", "--select-metric": "final_macro_f1",
        "--flag-source": "controlled_heuristic", "--save-selected-checkpoint": True,
        "--selected-checkpoint-filename": "selected_checkpoint.pt",
    }
    if options != expected:
        raise ValueError(f"{row['run']}: exact argv closure mismatch")
    forbidden = [name for name in options if any(fragment in name.lower()
                                                   for fragment in FORBIDDEN_FRAGMENTS)]
    if forbidden:
        raise ValueError(f"{row['run']}: prohibited options: {forbidden}")

def build(args: argparse.Namespace, manifest_dir: Path) -> dict[str, str]:
    repo = args.repo_root.resolve()
    parent = args.output_dir.resolve()
    trainer = (repo / TRAINER).resolve()
    data = (repo / DATA).resolve()
    if not repo.is_dir() or not trainer.is_file() or not data.is_file():
        raise FileNotFoundError("repository trainer or data absent")
    if manifest_dir.exists() or not parent.is_dir():
        raise FileExistsError("timestamped manifest target exists or parent is absent")
    if not re.fullmatch(r"[0-9a-f]{40}", args.current_git_commit or ""):
        raise ValueError("current commit must be lowercase full SHA")
    if args.stage196b1_runtime_git_commit != B1_RUNTIME_COMMIT:
        raise ValueError("historical Stage196-B1 runtime commit mismatch")
    head = git(repo, ["rev-parse", "HEAD"])
    if head != args.current_git_commit:
        raise ValueError("current commit differs from HEAD")
    git(repo, ["cat-file", "-e", f"{B1_RUNTIME_COMMIT}^{{commit}}"])
    git(repo, ["merge-base", "--is-ancestor", FRAMEGATE_ORIGIN_COMMIT, B1_RUNTIME_COMMIT])
    working_bytes = trainer.read_bytes()
    generator = (repo / GENERATOR).resolve()
    generator_bytes = generator.read_bytes()
    committed_generator_bytes = git(repo, ["show", f"{args.current_git_commit}:{GENERATOR.as_posix()}"], binary=True)
    if generator_bytes != committed_generator_bytes:
        raise ValueError("manifest-generator bytes differ from the supplied current commit")
    committed_bytes = git(repo, ["show", f"{args.current_git_commit}:{TRAINER.as_posix()}"],
                          binary=True)
    if working_bytes != committed_bytes:
        raise ValueError("trainer bytes differ from the supplied current commit")
    observed_data_sha = sha256(data)
    if observed_data_sha != DATA_SHA256:
        raise ValueError("main-data SHA mismatch")
    run_root = parent / RUN_ROOT
    rows: list[dict[str, Any]] = []
    expected_outputs: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            run = f"seed{seed}_{mode}"
            run_dir = run_root / run
            if run_dir.exists():
                raise FileExistsError(f"run output directory already exists: {run_dir}")
            outputs = run_outputs(run_dir)
            argv = build_argv(seed, mode, outputs)
            row = {
                "run": run, "seed": seed,
                "arm": "baseline" if mode == "joint" else "intervention",
                "mode": mode, "argv": argv,
                "command_argv": [sys.executable, str(trainer), *argv],
                "outputs": outputs, "execution_performed": False,
            }
            validate_command(row)
            rows.append(row)
            expected_outputs[run] = outputs
    if tuple(row["run"] for row in rows) != RUNS:
        raise ValueError("exact run ordering failed")
    normalized_rows = [normalized(row["argv"]) for row in rows]
    if len({json.dumps(value) for value in normalized_rows}) != 1:
        raise ValueError("commands differ beyond seed mode and output paths")
    if len({row["outputs"]["run_directory"] for row in rows}) != 6:
        raise ValueError("run output directories are not unique")
    historical_equivalence = {
        "unchanged_fields": [
            "seed", "split_seed", "data_path_and_sha", "architecture", "backbone",
            "model_name", "device", "epochs", "learning_rate", "freeze_encoder",
            "freeze_a_log", "frame_gradient_mode", "direct_framegate_bce_weight_1.0",
            "compatible_margin_zero", "class_weighting", "selection_metric",
            "max_length", "dev_ratio", "gradient_accumulation", "flag_source",
            "checkpoint_saving",
        ],
        "authorized_differences": [
            "output_root", "stage196b2p0_epoch_channel_observability_flag",
            "epoch_channel_sidecar_outputs",
        ],
        "prediction_equivalence_claimed_before_execution": False,
    }
    contract = {
        "ordered_runs": list(RUNS), "seeds": list(SEEDS), "split_seed": 174,
        "data_path": DATA.as_posix(), "data_sha256": observed_data_sha,
        "architecture": "v6b_minimal", "backbone": "mamba",
        "model_name": "state-spaces/mamba-130m-hf", "device": "cuda", "epochs": 20,
        "freeze_encoder": True, "freeze_a_log": True,
        "frame_direct_loss_active": True, "frame_direct_loss_weight": 1.0,
        "compatible_positive_margin_weight": 0.0,
        "compatible_positive_margin_logit": 0.0,
        "epoch_channel_file_count_per_run": 20, "epoch_channel_rows_per_file": 720,
        "required_fields": list(FIELDS), "no_extra_forward_pass": True,
        "training_semantics_changed": False, "gradient_semantics_changed": False,
        "checkpoint_selection_changed": False, "historical_equivalence": historical_equivalence,
    }
    source = {
        "passed": True, "trainer_path": str(trainer),
        "trainer_sha256": hashlib.sha256(working_bytes).hexdigest(),
        "manifest_generator_sha256": hashlib.sha256(generator_bytes).hexdigest(),
        "manifest_generator_bytes_equal_current_commit": True,
        "trainer_bytes_equal_current_commit": True,
        "data_sha256": observed_data_sha, "data_identity_passed": True,
        "current_head": head, "stage196b2p0_runtime_commit": args.current_git_commit,
        "stage196b1_runtime_commit": B1_RUNTIME_COMMIT,
        "framegate_implementation_origin_commit": FRAMEGATE_ORIGIN_COMMIT,
        "commit_roles_distinct": True,
    }
    manifest = {
        "stage": STAGE, "decision": READY, "blocking_reasons": [],
        "ordered_runs": list(RUNS), "run_root": str(run_root),
        "stage196b1_runtime_commit": B1_RUNTIME_COMMIT,
        "stage196b2p0_runtime_commit": args.current_git_commit,
        "framegate_implementation_origin_commit": FRAMEGATE_ORIGIN_COMMIT,
        "artifact_only_manifest_generation": True, "training_performed": False,
        "output_file_count": 7,
    }
    report = f"""# Stage196-B2-P0 epoch-channel observability manifest

## Decision

`{READY}`

## Run closure

Exact order: {", ".join(RUNS)}.

Each argv array preserves the Stage196-B1 configuration and adds only the new
observability flag and new output root. Every run expects 20 sidecars of 720 rows.

## No-extra-forward contract

The sidecars reuse the existing clean-dev epoch evaluation output. No extra model
forward, gradient, loss, optimizer, output, or checkpoint-selection change is authorized.

## Provenance

Original Stage196-B1 runtime: `{B1_RUNTIME_COMMIT}`.
Stage196-B2-P0 runtime: `{args.current_git_commit}`.
FrameGate implementation origin: `{FRAMEGATE_ORIGIN_COMMIT}`.

No prediction equivalence is claimed before execution.
"""
    return {
        OUTPUTS[0]: json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        OUTPUTS[2]: json.dumps(expected_outputs, indent=2, sort_keys=True) + "\n",
        OUTPUTS[3]: json.dumps(contract, indent=2, sort_keys=True) + "\n",
        OUTPUTS[4]: json.dumps(source, indent=2, sort_keys=True) + "\n",
        OUTPUTS[5]: "\n".join(RUNS) + "\n",
        OUTPUTS[6]: report,
    }

def publish(directory: Path, contents: dict[str, str]) -> None:
    if set(contents) != set(OUTPUTS):
        raise ValueError("exact seven-output closure failed")
    directory.mkdir(parents=False, exist_ok=False)
    for name in OUTPUTS:
        descriptor = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(contents[name])
    if {path.name for path in directory.iterdir()} != set(OUTPUTS):
        raise RuntimeError("published output closure mismatch")

def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = args.output_dir.resolve() / f"stage196b2p0_epoch_channel_observability_manifest_{stamp}"
    try:
        contents = build(args, target)
        code = 0
    except Exception as exc:
        error = {"stage": STAGE, "decision": BLOCKED,
                 "blocking_reasons": [f"{type(exc).__name__}: {exc}"],
                 "exception": traceback.format_exc(), "output_file_count": 7,
                 "training_performed": False}
        contents = {
            OUTPUTS[0]: json.dumps(error, indent=2, sort_keys=True) + "\n",
            OUTPUTS[1]: "", OUTPUTS[2]: "{}\n", OUTPUTS[3]: "{}\n",
            OUTPUTS[4]: json.dumps(error, indent=2, sort_keys=True) + "\n",
            OUTPUTS[5]: "\n".join(RUNS) + "\n",
            OUTPUTS[6]: f"# Stage196-B2-P0 manifest\n\n`{BLOCKED}`\n\n{error['blocking_reasons'][0]}\n",
        }
        code = 2
    publish(target, contents)
    print(json.dumps({"decision": json.loads(contents[OUTPUTS[0]])["decision"],
                      "output_dir": str(target), "output_files": list(OUTPUTS)}))
    return code

if __name__ == "__main__":
    raise SystemExit(main())

