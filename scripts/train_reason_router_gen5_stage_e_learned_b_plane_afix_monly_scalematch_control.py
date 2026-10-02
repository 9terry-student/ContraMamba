#!/usr/bin/env python3
"""Gen5 Stage E learned-B-plane AFIX-MONLY scale-matched diagnostic runner.

Design + implementation authority:
    1ea8d156c86a7f8ec6b9ad638632e46b54b0de38

Only --static-verify-only is authorized until a separate execution authority
and implementation freeze exist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contramamba.gen5_stage_e_learned_b_plane_afix_monly_scalematch_control import (  # noqa: E402
    ARM,
    EXPECTED_TRAINABLE_NUMEL,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneAFixMOnlyScaleMatchCorrection,
    install_learned_b_plane_afix_monly_scalematch_layer22_wrapper,
    load_seed_matched_afix_monly_scalematch_source,
    tensor_sha256,
    trainable_parameter_audit,
    zero_output_firewall,
)
from scripts import (  # noqa: E402
    train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control as afix,
)
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402
from scripts import train_reason_router_gen5_stage_e_causal_plane_bottleneck as stagee  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "1ea8d156c86a7f8ec6b9ad638632e46b54b0de38"
)
STATIC_SCALE_MODE_FREEZE_COMMIT = (
    "8abab776801eb106ddd97c39883a99f83257b833"
)
AFIX_EVIDENCE_FREEZE_COMMIT = (
    "d3d0f86fca9111ab19944f020c1efbb3d6b37d0a"
)
AFIX_IMPLEMENTATION_FREEZE_COMMIT = (
    "b1ab97600d47e0f86ca3a427befffb777a10829c"
)

AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_afix_monly_scalematch_"
    "design_implementation_authority_spec_candidate.md"
)
EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_afix_monly_scalematch_"
    "execution_authority_spec_candidate.md"
)

IMPLEMENTATION_PATHS = frozenset({
    "src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py",
    "scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py",
    "tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py",
})
FROZEN_PARENT_PATHS = frozenset({
    AUTHORITY_PATH,
    "reports/reason_router_gen5_stage_e_afix_monly_static_scale_mode_decomposition_report_candidate.md",
    "reports/reason_router_gen5_stage_e_learned_b_plane_afix_monly_validated_evidence_report_candidate.md",
    "src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_control.py",
    "scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py",
    "tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py",
})
PREFLIGHT_PREFIX = (
    "reports/reason_router_gen5_stage_e_bfree_afix_monly_scalematch_"
    "cuda_preflight_runs/"
)
RUN_PREFIX = (
    "reports/reason_router_gen5_stage_e_bfree_afix_monly_scalematch_"
    "control_runs/"
)

PRESSURE = "P0"
TRAIN_ROWS = stagee.TRAIN_ROWS
DEV_ROWS = stagee.DEV_ROWS
SPLIT_SEED = stagee.SPLIT_SEED
BACKBONE_STREAM_ROWS = stagee.BACKBONE_STREAM_ROWS
TOTAL_OPTIMIZER_STEPS = stagee.TOTAL_OPTIMIZER_STEPS
ORIGINAL_UNRESTRICTED_B_LEARNING_RATE = stagee.LEARNING_RATE
UNRESTRICTED_B_TRAINABLE_NUMEL = 24576 * 2
M_TRAINABLE_NUMEL = 2 * 2
SCALEMATCH_FACTOR = math.sqrt(
    UNRESTRICTED_B_TRAINABLE_NUMEL / M_TRAINABLE_NUMEL
)
LEARNING_RATE = ORIGINAL_UNRESTRICTED_B_LEARNING_RATE * SCALEMATCH_FACTOR
EXPECTED_SCALEMATCH_LEARNING_RATE = 0.11085125168440814
WEIGHT_DECAY = stagee.WEIGHT_DECAY
GRADIENT_CLIP_NORM = stagee.GRADIENT_CLIP_NORM
PARENT_CHECKPOINT_SHA256 = stagee.PARENT_CHECKPOINT_SHA256
ZERO_DEV_CE_P0 = stagee.ZERO_DEV_CE_P0
FREE_P0_GAIN_BY_SEED = dict(stagee.FREE_P0_GAIN_BY_SEED)

AFIX_MONLY_RECOVERY_BY_SEED = {
    6201: 0.063109275770733,
    6202: 0.0610331932890345,
    6203: 0.0648228579611734,
}
AFIX_MONLY_MEAN_RECOVERY = 0.0629884423403136
AINIT_RECOVERY_BY_SEED = dict(afix.AINIT_RECOVERY_BY_SEED)
AINIT_MEAN_RECOVERY = afix.AINIT_MEAN_RECOVERY
BFREE_RECOVERY_BY_SEED = dict(afix.BFREE_RECOVERY_BY_SEED)

MATRIX_GPU_COUNT = 2
MATRIX_WORKER_SEEDS = ((6201, 6203), (6202,))


class ScaleMatchRunnerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ScaleMatchRunnerError(message)


def _configure_afix_runtime() -> None:
    """Specialize frozen AFIX runtime helpers without modifying source files."""
    afix.IMPLEMENTATION_AUTHORITY_COMMIT = IMPLEMENTATION_AUTHORITY_COMMIT
    afix.STATIC_DECOMPOSITION_FREEZE_COMMIT = STATIC_SCALE_MODE_FREEZE_COMMIT
    afix.AINIT_EVIDENCE_FREEZE_COMMIT = AFIX_EVIDENCE_FREEZE_COMMIT
    afix.AUTHORITY_PATH = AUTHORITY_PATH
    afix.EXECUTION_AUTHORITY_PATH = EXECUTION_AUTHORITY_PATH
    afix.IMPLEMENTATION_PATHS = IMPLEMENTATION_PATHS
    afix.FROZEN_PARENT_PATHS = FROZEN_PARENT_PATHS
    afix.PREFLIGHT_PREFIX = PREFLIGHT_PREFIX
    afix.RUN_PREFIX = RUN_PREFIX
    afix.ARM = ARM
    afix.LEARNING_RATE = LEARNING_RATE
    afix.EXPECTED_TRAINABLE_NUMEL = EXPECTED_TRAINABLE_NUMEL
    afix.TRAINING_SEEDS = TRAINING_SEEDS
    afix.SOURCE_CORRECTIONS = SOURCE_CORRECTIONS
    afix.LearnedBPlaneAFixMOnlyCorrection = (
        LearnedBPlaneAFixMOnlyScaleMatchCorrection
    )
    afix.install_learned_b_plane_afix_monly_layer22_wrapper = (
        install_learned_b_plane_afix_monly_scalematch_layer22_wrapper
    )
    afix.load_seed_matched_afix_monly_source = (
        load_seed_matched_afix_monly_scalematch_source
    )
    afix.tensor_sha256 = tensor_sha256
    afix.trainable_parameter_audit = trainable_parameter_audit
    afix.zero_output_firewall = zero_output_firewall
    afix._validate_frozen_contract = _validate_frozen_contract
    afix._validate_execution_authority = _validate_execution_authority
    afix._checkpoint_payload = _checkpoint_payload
    afix._canonical_write = _canonical_write


_ORIGINAL_CANONICAL_WRITE = afix._canonical_write
_ORIGINAL_CHECKPOINT_PAYLOAD = afix._checkpoint_payload


def _validate_frozen_contract() -> None:
    require(TRAINING_SEEDS == (6201, 6202, 6203), "TRAINING_SEEDS")
    require(ARM == "E-BFREE-AFIX-MONLY-SCALEMATCH", "ARM")
    require(set(SOURCE_CORRECTIONS) == set(TRAINING_SEEDS), "SOURCE_SEEDS")
    require(TRAIN_ROWS == 3360, "TRAIN_ROWS")
    require(DEV_ROWS == 840, "DEV_ROWS")
    require(SPLIT_SEED == 16384, "SPLIT_SEED")
    require(PRESSURE == "P0", "PRESSURE")
    require(TOTAL_OPTIMIZER_STEPS == 20, "OPTIMIZER_STEPS")
    require(
        ORIGINAL_UNRESTRICTED_B_LEARNING_RATE == 0.001,
        "ORIGINAL_LEARNING_RATE",
    )
    require(
        UNRESTRICTED_B_TRAINABLE_NUMEL == 49152,
        "UNRESTRICTED_B_TRAINABLE_NUMEL",
    )
    require(M_TRAINABLE_NUMEL == 4, "M_TRAINABLE_NUMEL")
    require(
        SCALEMATCH_FACTOR == math.sqrt(12288),
        "SCALEMATCH_FACTOR",
    )
    require(
        LEARNING_RATE == EXPECTED_SCALEMATCH_LEARNING_RATE,
        "SCALEMATCH_LEARNING_RATE",
    )
    require(WEIGHT_DECAY == 0.0001, "WEIGHT_DECAY")
    require(GRADIENT_CLIP_NORM == 5.0, "GRADIENT_CLIP_NORM")
    require(EXPECTED_TRAINABLE_NUMEL == 4, "TRAINABLE_NUMEL")


def _validate_execution_authority(args: argparse.Namespace) -> None:
    require(
        args.execution_authority_commit is not None,
        "EXECUTION_AUTHORITY_COMMIT_REQUIRED",
    )
    require(
        args.implementation_freeze_commit is not None,
        "IMPLEMENTATION_FREEZE_COMMIT_REQUIRED",
    )
    require(
        afix.git_rc(
            "merge-base",
            "--is-ancestor",
            args.execution_authority_commit,
            args.expected_head,
        ) == 0,
        "EXECUTION_AUTHORITY_NOT_ANCESTOR",
    )
    authority = ROOT / EXECUTION_AUTHORITY_PATH
    require(authority.is_file(), "EXECUTION_AUTHORITY_FILE_MISSING")
    authority_blob = afix.git(
        "rev-parse",
        f"{args.execution_authority_commit}:{EXECUTION_AUTHORITY_PATH}",
    )
    live_blob = afix.git("rev-parse", f"HEAD:{EXECUTION_AUTHORITY_PATH}")
    require(authority_blob == live_blob, "EXECUTION_AUTHORITY_DRIFT")
    text = authority.read_text(encoding="utf-8")
    required_tokens = (
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL_DIAGNOSTIC",
        f"IMPLEMENTATION_FREEZE_COMMIT={args.implementation_freeze_commit}",
        "CUDA_PREFLIGHT_ALLOWED=YES",
        "TRAINING_ALLOWED=YES_EXACT_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL",
        "EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV",
        "BACKWARD_ALLOWED=YES_TRAINING_ONLY",
        "OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS_M_ONLY_SCALEMATCH_LR",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
    )
    for token in required_tokens:
        require(token in text, f"EXECUTION_AUTHORITY_TOKEN:{token}")


def _checkpoint_payload(
    *,
    wrapper: Any,
    args: argparse.Namespace,
    source_meta: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(
        wrapper=wrapper,
        args=args,
        source_meta=source_meta,
    )
    payload["schema_version"] = (
        "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_FINAL_CORRECTION_V1"
    )
    payload["arm"] = ARM
    payload["learning_rate"] = LEARNING_RATE
    payload["scalematch_factor"] = SCALEMATCH_FACTOR
    return payload


def _patch_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    patched = dict(value)
    schema = str(patched.get("schema_version", ""))
    if schema == "GEN5_STAGE_E_BFREE_AFIX_MONLY_CUDA_PREFLIGHT_V1":
        patched["schema_version"] = (
            "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_CUDA_PREFLIGHT_V1"
        )
        patched["result"] = (
            "PASS_GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_CUDA_PREFLIGHT"
        )
    elif schema == "GEN5_STAGE_E_BFREE_AFIX_MONLY_TRAINING_REPORT_V1":
        patched["schema_version"] = (
            "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_TRAINING_REPORT_V1"
        )
        patched["result"] = (
            "PASS_GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_TRAINING_CELL"
        )
        seed = int(patched["seed"])
        recovery = float(patched["recovery_vs_free_p0"])
        patched["frozen_afix_monly_recovery_seed_matched"] = (
            AFIX_MONLY_RECOVERY_BY_SEED[seed]
        )
        patched["delta_recovery_scalematch_minus_afix_monly"] = (
            recovery - AFIX_MONLY_RECOVERY_BY_SEED[seed]
        )
        patched["scalematch_factor"] = SCALEMATCH_FACTOR
        patched["unrestricted_b_trainable_numel_reference"] = (
            UNRESTRICTED_B_TRAINABLE_NUMEL
        )
        patched["M_trainable_numel"] = M_TRAINABLE_NUMEL
    elif schema == "GEN5_STAGE_E_BFREE_AFIX_MONLY_TRAINING_PROVENANCE_V1":
        patched["schema_version"] = (
            "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_TRAINING_PROVENANCE_V1"
        )
        patched["learning_rate"] = LEARNING_RATE
        patched["scalematch_factor"] = SCALEMATCH_FACTOR
    return patched


def _canonical_write(path: Path, value: Mapping[str, Any]) -> None:
    _ORIGINAL_CANONICAL_WRITE(path, _patch_artifact(value))


def run_static_verify(args: argparse.Namespace) -> dict[str, Any]:
    _configure_afix_runtime()
    repo = afix.authenticate_repo(
        args.expected_head,
        allow_opening_worktree=args.allow_opening_worktree,
    )
    _validate_frozen_contract()
    static = p3a.validate_static_artifacts()

    source_rows: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        q, a_free, source_meta = load_seed_matched_afix_monly_scalematch_source(
            ROOT, seed
        )
        correction = LearnedBPlaneAFixMOnlyScaleMatchCorrection(
            q_bfree=q,
            a_free=a_free,
            seed=seed,
        )
        require(
            tensor_sha256(correction.A_theta.weight)
            == source_meta["source_A_tensor_sha256"],
            f"STATIC_A_IDENTITY:{seed}",
        )
        require(
            not correction.A_theta.weight.requires_grad,
            f"STATIC_A_REQUIRES_GRAD:{seed}",
        )
        require(
            correction.M_theta.weight.requires_grad,
            f"STATIC_M_NOT_TRAINABLE:{seed}",
        )
        firewall = zero_output_firewall(correction)
        audit = trainable_parameter_audit(correction)
        source_rows[str(seed)] = {
            "source": source_meta,
            "step_zero_firewall": firewall,
            "parameter_audit": audit,
        }

    report = {
        "schema_version": (
            "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_STATIC_VERIFY_V1"
        ),
        "result": (
            "PASS_GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_STATIC_VERIFY"
        ),
        "execution_head": args.expected_head,
        "repository": repo,
        "static_tree_sha256": static["tree"]["tree_sha256"],
        "train_order_sha256": p3a.row_order_sha256(static["train_rows"]),
        "dev_order_sha256": p3a.row_order_sha256(static["dev_rows"]),
        "seeds": list(TRAINING_SEEDS),
        "arm": ARM,
        "pressure": PRESSURE,
        "seed_matched_sources": source_rows,
        "original_unrestricted_b_learning_rate": (
            ORIGINAL_UNRESTRICTED_B_LEARNING_RATE
        ),
        "unrestricted_b_trainable_numel": UNRESTRICTED_B_TRAINABLE_NUMEL,
        "M_trainable_numel": M_TRAINABLE_NUMEL,
        "scalematch_factor": SCALEMATCH_FACTOR,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "expected_trainable_numel": EXPECTED_TRAINABLE_NUMEL,
        "A_frozen": True,
        "M_only_trainable": True,
        "parent_checkpoint_loaded": False,
        "parent_model_instantiated": False,
        "model_forward_count": 0,
        "cuda_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "training_executed": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }

    print("GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"STATIC_TREE_SHA256={static['tree']['tree_sha256']}")
    print(f"ARM={ARM}")
    print("SEEDS=6201,6202,6203")
    print("PRESSURE=P0")
    print(
        f"ORIGINAL_UNRESTRICTED_B_LEARNING_RATE="
        f"{ORIGINAL_UNRESTRICTED_B_LEARNING_RATE:.17g}"
    )
    print(
        f"UNRESTRICTED_B_TRAINABLE_NUMEL={UNRESTRICTED_B_TRAINABLE_NUMEL}"
    )
    print(f"M_TRAINABLE_NUMEL={M_TRAINABLE_NUMEL}")
    print(f"SCALEMATCH_FACTOR={SCALEMATCH_FACTOR:.17g}")
    print(f"LEARNING_RATE={LEARNING_RATE:.17g}")
    print("TRAINABLE_TENSOR_NAMES=M_theta.weight")
    print("TRAINABLE_TENSOR_COUNT=1")
    print("TRAINABLE_NUMEL=4")
    print("A_REQUIRES_GRAD=False")
    print("M_REQUIRES_GRAD=True")
    for seed in TRAINING_SEEDS:
        source = source_rows[str(seed)]["source"]
        print(
            f"SEED={seed} "
            f"SOURCE_SHA256={source['source_file_sha256']} "
            f"A_FREE_SHA256={source['source_A_tensor_sha256']} "
            f"B_FREE_SHA256={source['source_B_tensor_sha256']} "
            f"Q_SHA256={source['Q_tensor_sha256']} "
            f"R_SHA256={source['R_tensor_sha256']} "
            f"A_ROW_RANK={source['source_A_row_rank']} "
            f"B_RANK={source['source_B_rank']} "
            f"QR_RECON_REL={source['qr_reconstruction_relative']:.12g} "
            f"REPRESENTABILITY_REL="
            f"{source['factorized_operator_representability_relative']:.12g}"
        )
    print("STEP0_CORRECTION_EXACT_ZERO=True")
    print("PARENT_CHECKPOINT_LOADED=False")
    print("PARENT_MODEL_INSTANTIATED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("TRAINING_EXECUTED=False")
    print("TASK_EVALUATION_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    return report


def run_cuda_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _configure_afix_runtime()
    report = afix.run_cuda_preflight(args)
    patched = _patch_artifact(report)
    print("GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_CUDA_PREFLIGHT_PASS")
    return patched


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    _configure_afix_runtime()
    report = afix.run_cell(args)
    patched = _patch_artifact(report)
    print(
        f"GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_CELL_PASS "
        f"seed={args.seed} "
        f"recovery={float(patched['recovery_vs_free_p0']):.12g} "
        f"delta_vs_afix="
        f"{float(patched['delta_recovery_scalematch_minus_afix_monly']):.12g}"
    )
    return patched


def _cell_command(
    args: argparse.Namespace,
    *,
    seed: int,
    scratch_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-cell",
        "--expected-head", args.expected_head,
        "--implementation-freeze-commit", str(args.implementation_freeze_commit),
        "--execution-authority-commit", str(args.execution_authority_commit),
        "--seed", str(seed),
        "--checkpoint", str(args.checkpoint),
        "--output-root", str(scratch_root),
    ]
    if args.model_snapshot is not None:
        command += ["--model-snapshot", str(args.model_snapshot)]
    if args.tokenizer_snapshot is not None:
        command += ["--tokenizer-snapshot", str(args.tokenizer_snapshot)]
    return command


def _run_worker(
    args: argparse.Namespace,
    *,
    worker_index: int,
    seeds: Sequence[int],
    scratch_root: Path,
    log_path: Path,
) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
    with log_path.open("w", encoding="utf-8") as log:
        for seed in seeds:
            completed = subprocess.run(
                _cell_command(args, seed=seed, scratch_root=scratch_root),
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                return int(completed.returncode)
    return 0


def _namespace_from_dict(values: Mapping[str, Any]) -> argparse.Namespace:
    converted = dict(values)
    for key in (
        "tokenizer_snapshot",
        "model_snapshot",
        "checkpoint",
        "preflight_output",
        "output_root",
    ):
        if converted.get(key) is not None:
            converted[key] = Path(converted[key])
    return argparse.Namespace(**converted)


def _matrix_worker_entry(
    worker_index: int,
    seeds: Sequence[int],
    scratch_root: str,
    log_path: str,
    args_dict: Mapping[str, Any],
) -> int:
    return _run_worker(
        _namespace_from_dict(args_dict),
        worker_index=worker_index,
        seeds=seeds,
        scratch_root=Path(scratch_root),
        log_path=Path(log_path),
    )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    _configure_afix_runtime()
    afix.authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    gpu_meta = stagee._validate_two_t4s()

    output_root = Path(args.output_root)
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")
    scratch_parent = Path(
        tempfile.mkdtemp(prefix="contramamba_stage_e_scalematch_")
    )
    worker_roots = [
        scratch_parent / f"worker{i}" for i in range(MATRIX_GPU_COUNT)
    ]
    worker_logs = [
        scratch_parent / f"worker{i}.log" for i in range(MATRIX_GPU_COUNT)
    ]
    for path in worker_roots:
        path.mkdir(parents=True)

    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    processes = []
    for worker_index, seeds in enumerate(MATRIX_WORKER_SEEDS):
        command = [
            sys.executable,
            "-c",
            (
                "from scripts."
                "train_reason_router_gen5_stage_e_learned_b_plane_"
                "afix_monly_scalematch_control "
                "import _matrix_worker_entry; "
                "raise SystemExit(_matrix_worker_entry("
                f"{worker_index!r}, {list(seeds)!r}, "
                f"{str(worker_roots[worker_index])!r}, "
                f"{str(worker_logs[worker_index])!r}, "
                f"{args_payload!r}))"
            ),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        processes.append(
            subprocess.Popen(command, cwd=ROOT, env=env, text=True)
        )

    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        output_root.mkdir(parents=True, exist_ok=False)
        fail_dir = output_root / "failed_worker_logs"
        fail_dir.mkdir()
        for src in worker_logs:
            if src.exists():
                shutil.copy2(src, fail_dir / src.name)
        raise ScaleMatchRunnerError(
            f"MATRIX_WORKER_FAILURE:{return_codes}"
        )

    output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    for worker_index, worker_root in enumerate(worker_roots):
        for seed in MATRIX_WORKER_SEEDS[worker_index]:
            source = worker_root / f"seed{seed}" / ARM
            require(source.is_dir(), f"WORKER_CELL_MISSING:{seed}")
            destination = output_root / f"seed{seed}" / ARM
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            reports.append(
                json.loads(
                    (destination / "training_report.json")
                    .read_text(encoding="utf-8")
                )
            )
    reports.sort(key=lambda row: int(row["seed"]))
    require(len(reports) == 3, "MATRIX_CELL_COUNT")

    mean_recovery = sum(
        float(row["recovery_vs_free_p0"]) for row in reports
    ) / 3.0
    mean_delta_afix = sum(
        float(row["delta_recovery_scalematch_minus_afix_monly"])
        for row in reports
    ) / 3.0

    summary = {
        "schema_version": (
            "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_MATRIX_SUMMARY_V1"
        ),
        "result": (
            "PASS_GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL_MATRIX"
        ),
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "gpu_meta": gpu_meta,
        "worker_seeds": [list(x) for x in MATRIX_WORKER_SEEDS],
        "cell_count": 3,
        "cells": reports,
        "learning_rate": LEARNING_RATE,
        "scalematch_factor": SCALEMATCH_FACTOR,
        "unrestricted_b_trainable_numel_reference": (
            UNRESTRICTED_B_TRAINABLE_NUMEL
        ),
        "M_trainable_numel": M_TRAINABLE_NUMEL,
        "mean_recovery_E_BFREE_AFIX_MONLY_SCALEMATCH": mean_recovery,
        "frozen_mean_recovery_E_BFREE_AFIX_MONLY": (
            AFIX_MONLY_MEAN_RECOVERY
        ),
        "mean_delta_SCALEMATCH_minus_AFIX_MONLY": mean_delta_afix,
        "frozen_mean_recovery_E_BFREE_AINIT": AINIT_MEAN_RECOVERY,
        "training_executed": True,
        "backward_executed": True,
        "optimizer_step_count": 3 * TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    summary_path = output_root / "matrix_summary.json"
    _canonical_write(summary_path, summary)

    provenance = {
        "schema_version": (
            "GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_MATRIX_PROVENANCE_V1"
        ),
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "matrix_summary_sha256": stagee.sha256_file(summary_path),
        "cell_count": 3,
        "learning_rate": LEARNING_RATE,
        "scalematch_factor": SCALEMATCH_FACTOR,
        "optimizer_step_count": 3 * TOTAL_OPTIMIZER_STEPS,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _canonical_write(output_root / "run_provenance.json", provenance)
    shutil.rmtree(scratch_parent)

    print("GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL_MATRIX_PASS")
    print("CELLS=3")
    print("GPU_WORKERS=2")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("PRESSURE=P0")
    print("TRAINABLE_NUMEL_PER_CELL=4")
    print("A_FROZEN=True")
    print(f"LEARNING_RATE={LEARNING_RATE:.17g}")
    print(f"SCALEMATCH_FACTOR={SCALEMATCH_FACTOR:.17g}")
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_OPTIMIZER_STEPS=60")
    print("TRAINING_EXECUTED=True")
    print("BACKWARD_EXECUTED=True")
    print("TASK_EVALUATION_EXECUTED=True")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(
        f"MEAN_RECOVERY_E_BFREE_AFIX_MONLY_SCALEMATCH="
        f"{mean_recovery:.15g}"
    )
    print(
        f"FROZEN_MEAN_RECOVERY_E_BFREE_AFIX_MONLY="
        f"{AFIX_MONLY_MEAN_RECOVERY:.15g}"
    )
    print(
        f"MEAN_DELTA_SCALEMATCH_MINUS_AFIX_MONLY="
        f"{mean_delta_afix:.15g}"
    )
    print(f"SUMMARY={summary_path}")
    print(f"PROVENANCE={output_root / 'run_provenance.json'}")
    return summary


build_parser = afix.build_parser
validate_mode_args = afix.validate_mode_args


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_mode_args(args)
    if args.static_verify_only:
        run_static_verify(args)
    elif args.cuda_preflight_only:
        run_cuda_preflight(args)
    elif args.run_cell:
        run_cell(args)
    else:
        run_matrix(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
