#!/usr/bin/env python3
"""Gen5 Stage E learned-B-plane A-initialization diagnostic runner.

Design + implementation authority:
    c24bc199b6b8fb94002c0563535ff7b5284794b2

Only --static-verify-only is authorized before a later implementation freeze
and separate execution authority.
"""

from __future__ import annotations

import argparse
import json
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

from contramamba.gen5_stage_e_causal_plane_bottleneck import (  # noqa: E402
    EXPECTED_TRAINABLE_NUMEL,
)
from contramamba.gen5_stage_e_learned_b_plane_ainit_control import (  # noqa: E402
    ARM,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneAInitCorrection,
    install_learned_b_plane_ainit_layer22_wrapper,
    load_seed_matched_ainit_source,
    tensor_sha256,
    trainable_parameter_audit,
    zero_output_firewall,
)
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402
from scripts import train_reason_router_gen5_stage_e_causal_plane_bottleneck as stagee  # noqa: E402
from scripts import train_reason_router_gen5_stage_e_learned_b_plane_positive_control as bfree_runner  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "c24bc199b6b8fb94002c0563535ff7b5284794b2"
)
BFREE_EVIDENCE_FREEZE_COMMIT = (
    "a17006164d6fc73de64cb138b12b6a7975750b4e"
)
AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_learned_b_plane_ainit_"
    "design_implementation_authority_spec_candidate.md"
)

IMPLEMENTATION_PATHS = frozenset({
    "src/contramamba/gen5_stage_e_learned_b_plane_ainit_control.py",
    "scripts/train_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py",
    "tests/test_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py",
})
FROZEN_PARENT_PATHS = frozenset({
    AUTHORITY_PATH,
    "src/contramamba/gen5_stage_e_learned_b_plane_positive_control.py",
    "scripts/train_reason_router_gen5_stage_e_learned_b_plane_positive_control.py",
    "tests/test_reason_router_gen5_stage_e_learned_b_plane_positive_control.py",
    "reports/reason_router_gen5_stage_e_learned_b_plane_positive_control_"
    "validated_evidence_report_candidate.md",
    "reports/reason_router_gen5_stage_e_bfree_positive_control_runs/"
    "gen5-stagee-bfree-three-cell-467148d-r1/matrix_summary.json",
    "reports/reason_router_gen5_stage_e_bfree_positive_control_runs/"
    "gen5-stagee-bfree-three-cell-467148d-r1/run_provenance.json",
})

EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_learned_b_plane_ainit_"
    "execution_authority_spec_candidate.md"
)
PREFLIGHT_PREFIX = (
    "reports/reason_router_gen5_stage_e_bfree_ainit_cuda_preflight_runs/"
)
RUN_PREFIX = (
    "reports/reason_router_gen5_stage_e_bfree_ainit_control_runs/"
)

PRESSURE = "P0"
TRAIN_ROWS = stagee.TRAIN_ROWS
DEV_ROWS = stagee.DEV_ROWS
SPLIT_SEED = stagee.SPLIT_SEED
TOTAL_OPTIMIZER_STEPS = stagee.TOTAL_OPTIMIZER_STEPS
LEARNING_RATE = stagee.LEARNING_RATE
WEIGHT_DECAY = stagee.WEIGHT_DECAY
GRADIENT_CLIP_NORM = stagee.GRADIENT_CLIP_NORM
PARENT_CHECKPOINT_SHA256 = stagee.PARENT_CHECKPOINT_SHA256
ZERO_DEV_CE_P0 = stagee.ZERO_DEV_CE_P0
FREE_P0_GAIN_BY_SEED = dict(stagee.FREE_P0_GAIN_BY_SEED)
BFREE_RECOVERY_BY_SEED = {
    6201: 0.0690419274517625,
    6202: 0.0737218360466545,
    6203: 0.0715115026716749,
}
BFREE_MEAN_RECOVERY = 0.07142508872336398
MATRIX_GPU_COUNT = 2
MATRIX_WORKER_SEEDS = ((6201, 6203), (6202,))


class AInitRunnerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AInitRunnerError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AInitRunnerError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _status_paths() -> set[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    result: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        require(len(line) >= 4, f"MALFORMED_STATUS:{line!r}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result.add(path)
    return result


def _post_authority_path_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        normalized in IMPLEMENTATION_PATHS
        or normalized == EXECUTION_AUTHORITY_PATH
        or normalized.startswith(PREFLIGHT_PREFIX)
        or normalized.startswith(RUN_PREFIX)
    )


def authenticate_repo(
    expected_head: str,
    *,
    allow_opening_worktree: bool = False,
    implementation_freeze_commit: str | None = None,
) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(head == expected_head, f"HEAD:{head}")
    require(branch in (EXPECTED_BRANCH, ""), f"BRANCH:{branch}")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            IMPLEMENTATION_AUTHORITY_COMMIT,
            head,
        ) == 0,
        "IMPLEMENTATION_AUTHORITY_NOT_ANCESTOR",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            BFREE_EVIDENCE_FREEZE_COMMIT,
            head,
        ) == 0,
        "BFREE_EVIDENCE_NOT_ANCESTOR",
    )

    status = _status_paths()
    if allow_opening_worktree:
        require(
            status <= IMPLEMENTATION_PATHS,
            f"UNAUTHORIZED_WORKTREE_PATHS:{sorted(status)}",
        )
    else:
        require(not status, f"WORKTREE_NOT_CLEAN:{sorted(status)}")

    for path in sorted(FROZEN_PARENT_PATHS):
        authority_blob = git(
            "rev-parse",
            f"{IMPLEMENTATION_AUTHORITY_COMMIT}:{path}",
        )
        live_blob = git("rev-parse", f"HEAD:{path}")
        require(
            authority_blob == live_blob,
            f"FROZEN_PARENT_DRIFT:{path}",
        )

    if head != IMPLEMENTATION_AUTHORITY_COMMIT:
        changed = {
            line.strip().replace("\\", "/")
            for line in git(
                "diff",
                "--name-only",
                f"{IMPLEMENTATION_AUTHORITY_COMMIT}..{head}",
            ).splitlines()
            if line.strip()
        }
        unauthorized = {
            path for path in changed
            if not _post_authority_path_allowed(path)
        }
        require(
            not unauthorized,
            f"POST_AUTHORITY_SCOPE:{sorted(unauthorized)}",
        )

    if implementation_freeze_commit is not None:
        require(
            git_rc(
                "merge-base",
                "--is-ancestor",
                implementation_freeze_commit,
                head,
            ) == 0,
            "IMPLEMENTATION_FREEZE_NOT_ANCESTOR",
        )
        for path in sorted(IMPLEMENTATION_PATHS):
            frozen_blob = git(
                "rev-parse",
                f"{implementation_freeze_commit}:{path}",
            )
            live_blob = git("rev-parse", f"HEAD:{path}")
            require(
                frozen_blob == live_blob,
                f"IMPLEMENTATION_DRIFT:{path}",
            )

    return {
        "branch": branch,
        "head": head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "implementation_freeze_commit": implementation_freeze_commit,
    }


def _validate_frozen_contract() -> None:
    require(TRAINING_SEEDS == (6201, 6202, 6203), "TRAINING_SEEDS")
    require(ARM == "E-BFREE-AINIT", "ARM")
    require(set(SOURCE_CORRECTIONS) == set(TRAINING_SEEDS), "SOURCE_SEEDS")
    require(TRAIN_ROWS == 3360, "TRAIN_ROWS")
    require(DEV_ROWS == 840, "DEV_ROWS")
    require(SPLIT_SEED == 16384, "SPLIT_SEED")
    require(PRESSURE == "P0", "PRESSURE")
    require(TOTAL_OPTIMIZER_STEPS == 20, "OPTIMIZER_STEPS")
    require(LEARNING_RATE == 0.001, "LEARNING_RATE")
    require(WEIGHT_DECAY == 0.0001, "WEIGHT_DECAY")
    require(GRADIENT_CLIP_NORM == 5.0, "GRADIENT_CLIP_NORM")
    require(EXPECTED_TRAINABLE_NUMEL == 1540, "TRAINABLE_NUMEL")
    require(
        BFREE_RECOVERY_BY_SEED == {
            6201: 0.0690419274517625,
            6202: 0.0737218360466545,
            6203: 0.0715115026716749,
        },
        "BFREE_REFERENCE",
    )


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
        git_rc(
            "merge-base",
            "--is-ancestor",
            args.execution_authority_commit,
            args.expected_head,
        ) == 0,
        "EXECUTION_AUTHORITY_NOT_ANCESTOR",
    )
    authority = ROOT / EXECUTION_AUTHORITY_PATH
    require(authority.is_file(), "EXECUTION_AUTHORITY_FILE_MISSING")
    authority_blob = git(
        "rev-parse",
        f"{args.execution_authority_commit}:{EXECUTION_AUTHORITY_PATH}",
    )
    live_blob = git("rev-parse", f"HEAD:{EXECUTION_AUTHORITY_PATH}")
    require(
        authority_blob == live_blob,
        "EXECUTION_AUTHORITY_DRIFT",
    )
    text = authority.read_text(encoding="utf-8")
    required_tokens = (
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_BFREE_AINIT_THREE_CELL_DIAGNOSTIC",
        f"IMPLEMENTATION_FREEZE_COMMIT={args.implementation_freeze_commit}",
        "CUDA_PREFLIGHT_ALLOWED=YES",
        "TRAINING_ALLOWED=YES_EXACT_STAGE_E_BFREE_AINIT_THREE_CELL",
        "EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV",
        "BACKWARD_ALLOWED=YES_TRAINING_ONLY",
        "OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
    )
    for token in required_tokens:
        require(token in text, f"EXECUTION_AUTHORITY_TOKEN:{token}")


def _canonical_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(stagee.canonical_json_bytes(dict(value)) + b"\n")


def run_static_verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = authenticate_repo(
        args.expected_head,
        allow_opening_worktree=args.allow_opening_worktree,
    )
    _validate_frozen_contract()
    static = p3a.validate_static_artifacts()

    source_rows: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        q, a_free, source_meta = load_seed_matched_ainit_source(ROOT, seed)
        correction = LearnedBPlaneAInitCorrection(
            q_bfree=q,
            a_free=a_free,
            seed=seed,
        )
        require(
            tensor_sha256(correction.A_theta.weight)
            == source_meta["source_A_tensor_sha256"],
            f"AINIT_STATIC_A_IDENTITY:{seed}",
        )
        firewall = zero_output_firewall(correction)
        audit = trainable_parameter_audit(correction)
        source_rows[str(seed)] = {
            "source": source_meta,
            "step_zero_firewall": firewall,
            "parameter_audit": audit,
        }

    report = {
        "schema_version": "GEN5_STAGE_E_BFREE_AINIT_STATIC_VERIFY_V1",
        "result": "PASS_GEN5_STAGE_E_BFREE_AINIT_STATIC_VERIFY",
        "execution_head": args.expected_head,
        "repository": repo,
        "static_tree_sha256": static["tree"]["tree_sha256"],
        "train_order_sha256": p3a.row_order_sha256(static["train_rows"]),
        "dev_order_sha256": p3a.row_order_sha256(static["dev_rows"]),
        "seeds": list(TRAINING_SEEDS),
        "arm": ARM,
        "pressure": PRESSURE,
        "seed_matched_sources": source_rows,
        "expected_trainable_numel": EXPECTED_TRAINABLE_NUMEL,
        "source_correction_checkpoint_count": 3,
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

    print("GEN5_STAGE_E_BFREE_AINIT_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"STATIC_TREE_SHA256={static['tree']['tree_sha256']}")
    print("ARM=E-BFREE-AINIT")
    print("SEEDS=6201,6202,6203")
    print("PRESSURE=P0")
    print("EXPECTED_TRAINABLE_NUMEL=1540")
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
    print("M_INIT_EXACT_ZERO=True")
    print("STEP0_CORRECTION_EXACT_ZERO=True")
    print("SOURCE_CORRECTION_CHECKPOINT_COUNT=3")
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


def _plane_adapter(
    repo_root: str | Path,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    q, _a, metadata = load_seed_matched_ainit_source(repo_root, seed)
    return q, metadata


def _installer_adapter(
    model: torch.nn.Module,
    *,
    q_bfree: torch.Tensor,
    seed: int,
    **kwargs: Any,
):
    q, a_free, metadata = load_seed_matched_ainit_source(ROOT, seed)
    require(
        tensor_sha256(q) == tensor_sha256(q_bfree),
        f"AINIT_RUNTIME_Q_IDENTITY:{seed}",
    )
    wrapper = install_learned_b_plane_ainit_layer22_wrapper(
        model,
        q_bfree=q,
        a_free=a_free,
        seed=seed,
        **kwargs,
    )
    require(
        tensor_sha256(wrapper.correction.A_theta.weight)
        == metadata["source_A_tensor_sha256"],
        f"AINIT_RUNTIME_A_IDENTITY:{seed}",
    )
    return wrapper


def _checkpoint_payload(
    *,
    wrapper: Any,
    args: argparse.Namespace,
    source_meta: Mapping[str, Any],
) -> dict[str, Any]:
    a = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    m = wrapper.correction.M_theta.weight.detach().cpu().contiguous()
    return {
        "schema_version": "GEN5_STAGE_E_BFREE_AINIT_FINAL_CORRECTION_V1",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "seed": args.seed,
        "arm": ARM,
        "pressure": PRESSURE,
        "source_plane": dict(source_meta),
        "initialization": "SEED_MATCHED_PHASE3A_P0_FINAL_A",
        "state_dict": {
            "A_theta.weight": a,
            "M_theta.weight": m,
        },
        "tensor_sha256": {
            "A_theta.weight": tensor_sha256(a),
            "M_theta.weight": tensor_sha256(m),
        },
    }


def _patch_bfree_runtime() -> None:
    bfree_runner.ARM = ARM
    bfree_runner.authenticate_repo = authenticate_repo
    bfree_runner._validate_execution_authority = _validate_execution_authority
    bfree_runner._validate_frozen_contract = _validate_frozen_contract
    bfree_runner.load_seed_matched_learned_plane = _plane_adapter
    bfree_runner.install_learned_b_plane_layer22_wrapper = _installer_adapter
    bfree_runner._checkpoint_payload = _checkpoint_payload


def _postprocess_preflight(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value["schema_version"] = "GEN5_STAGE_E_BFREE_AINIT_CUDA_PREFLIGHT_V1"
    value["result"] = "PASS_GEN5_STAGE_E_BFREE_AINIT_CUDA_PREFLIGHT"
    value["arm"] = ARM
    value["initialization"] = "SEED_MATCHED_PHASE3A_P0_FINAL_A"
    _canonical_write(path, value)
    return value


def run_cuda_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _patch_bfree_runtime()
    report = bfree_runner.run_cuda_preflight(args)
    value = _postprocess_preflight(Path(args.preflight_output), report)
    print("GEN5_STAGE_E_BFREE_AINIT_CUDA_PREFLIGHT_PASS")
    print(f"SEED={args.seed}")
    print("ARM=E-BFREE-AINIT")
    print("INITIALIZATION=SEED_MATCHED_PHASE3A_P0_FINAL_A")
    return value


def _postprocess_cell(
    cell_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    report_path = cell_dir / "training_report.json"
    provenance_path = cell_dir / "run_provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "GEN5_STAGE_E_BFREE_AINIT_TRAINING_REPORT_V1"
    report["result"] = "PASS_GEN5_STAGE_E_BFREE_AINIT_TRAINING_CELL"
    report["arm"] = ARM
    report["initialization"] = "SEED_MATCHED_PHASE3A_P0_FINAL_A"
    report["frozen_bfree_recovery_seed_matched"] = BFREE_RECOVERY_BY_SEED[seed]
    report["delta_recovery_ainit_minus_bfree"] = (
        float(report["recovery_vs_free_p0"]) - BFREE_RECOVERY_BY_SEED[seed]
    )
    _canonical_write(report_path, report)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["schema_version"] = (
        "GEN5_STAGE_E_BFREE_AINIT_TRAINING_PROVENANCE_V1"
    )
    provenance["status"] = "PASS"
    provenance["arm"] = ARM
    provenance["initialization"] = "SEED_MATCHED_PHASE3A_P0_FINAL_A"
    provenance["Q_tensor_sha256"] = report["source_plane"]["Q_tensor_sha256"]
    provenance["source_A_tensor_sha256"] = (
        report["source_plane"]["source_A_tensor_sha256"]
    )
    provenance["training_report_sha256"] = stagee.sha256_file(report_path)
    _canonical_write(provenance_path, provenance)
    return report


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    _patch_bfree_runtime()
    bfree_runner.run_cell(args)
    cell_dir = Path(args.output_root) / f"seed{args.seed}" / ARM
    report = _postprocess_cell(cell_dir, seed=int(args.seed))
    print(
        f"GEN5_STAGE_E_BFREE_AINIT_CELL_PASS seed={args.seed} "
        f"dev_ce={float(report['dev_final_3way_ce']):.12g} "
        f"gain={float(report['gain_vs_zero']):.12g} "
        f"recovery={float(report['recovery_vs_free_p0']):.12g} "
        f"delta_vs_bfree="
        f"{float(report['delta_recovery_ainit_minus_bfree']):.12g}"
    )
    return report


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
                _cell_command(
                    args,
                    seed=seed,
                    scratch_root=scratch_root,
                ),
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
    authenticate_repo(
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
        tempfile.mkdtemp(prefix="contramamba_stage_e_bfree_ainit_")
    )
    worker_roots = [
        scratch_parent / f"worker{i}"
        for i in range(MATRIX_GPU_COUNT)
    ]
    worker_logs = [
        scratch_parent / f"worker{i}.log"
        for i in range(MATRIX_GPU_COUNT)
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
                "train_reason_router_gen5_stage_e_learned_b_plane_ainit_control "
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
        raise AInitRunnerError(
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
    mean_delta = sum(
        float(row["delta_recovery_ainit_minus_bfree"])
        for row in reports
    ) / 3.0
    summary = {
        "schema_version": "GEN5_STAGE_E_BFREE_AINIT_MATRIX_SUMMARY_V1",
        "result": "PASS_GEN5_STAGE_E_BFREE_AINIT_THREE_CELL_MATRIX",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "gpu_meta": gpu_meta,
        "worker_seeds": [list(x) for x in MATRIX_WORKER_SEEDS],
        "cell_count": 3,
        "cells": reports,
        "mean_recovery_E_BFREE_AINIT": mean_recovery,
        "frozen_mean_recovery_E_BFREE": BFREE_MEAN_RECOVERY,
        "mean_delta_AINIT_minus_BFREE": mean_delta,
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
        "schema_version": "GEN5_STAGE_E_BFREE_AINIT_MATRIX_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "matrix_summary_sha256": stagee.sha256_file(summary_path),
        "cell_count": 3,
        "optimizer_step_count": 3 * TOTAL_OPTIMIZER_STEPS,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _canonical_write(output_root / "run_provenance.json", provenance)
    shutil.rmtree(scratch_parent)

    print("GEN5_STAGE_E_BFREE_AINIT_THREE_CELL_MATRIX_PASS")
    print("CELLS=3")
    print("GPU_WORKERS=2")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("PRESSURE=P0")
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_OPTIMIZER_STEPS=60")
    print("TRAINING_EXECUTED=True")
    print("BACKWARD_EXECUTED=True")
    print("TASK_EVALUATION_EXECUTED=True")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"MEAN_RECOVERY_E_BFREE_AINIT={mean_recovery:.15g}")
    print(f"FROZEN_MEAN_RECOVERY_E_BFREE={BFREE_MEAN_RECOVERY:.15g}")
    print(f"MEAN_DELTA_AINIT_MINUS_BFREE={mean_delta:.15g}")
    print(f"SUMMARY={summary_path}")
    print(f"PROVENANCE={output_root / 'run_provenance.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--cuda-preflight-only", action="store_true")
    modes.add_argument("--run-cell", action="store_true")
    modes.add_argument("--run-matrix", action="store_true")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--allow-opening-worktree", action="store_true")
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--execution-authority-commit")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(
            args.implementation_freeze_commit is None,
            "STATIC_IMPLEMENTATION_FREEZE_FORBIDDEN",
        )
        require(
            args.execution_authority_commit is None,
            "STATIC_EXECUTION_AUTHORITY_FORBIDDEN",
        )
        require(args.seed is None, "STATIC_SEED_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(
            args.preflight_output is None,
            "STATIC_PREFLIGHT_OUTPUT_FORBIDDEN",
        )
        require(args.output_root is None, "STATIC_OUTPUT_ROOT_FORBIDDEN")
        require(
            args.model_snapshot is None,
            "STATIC_MODEL_SNAPSHOT_FORBIDDEN",
        )
        require(
            args.tokenizer_snapshot is None,
            "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN",
        )
        return

    require(
        not args.allow_opening_worktree,
        "RUNTIME_OPENING_WORKTREE_FORBIDDEN",
    )
    require(
        args.implementation_freeze_commit is not None,
        "RUNTIME_IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(
        args.execution_authority_commit is not None,
        "RUNTIME_EXECUTION_AUTHORITY_REQUIRED",
    )
    require(args.checkpoint is not None, "RUNTIME_CHECKPOINT_REQUIRED")

    if args.cuda_preflight_only:
        require(args.seed in TRAINING_SEEDS, "PREFLIGHT_SEED_REQUIRED")
        require(
            args.preflight_output is not None,
            "PREFLIGHT_OUTPUT_REQUIRED",
        )
        require(
            args.output_root is None,
            "PREFLIGHT_OUTPUT_ROOT_FORBIDDEN",
        )
    elif args.run_cell:
        require(args.seed in TRAINING_SEEDS, "RUN_CELL_SEED_REQUIRED")
        require(args.output_root is not None, "RUN_CELL_OUTPUT_ROOT_REQUIRED")
        require(
            args.preflight_output is None,
            "RUN_CELL_PREFLIGHT_OUTPUT_FORBIDDEN",
        )
    else:
        require(args.seed is None, "MATRIX_SEED_FORBIDDEN")
        require(args.output_root is not None, "MATRIX_OUTPUT_ROOT_REQUIRED")
        require(
            args.preflight_output is None,
            "MATRIX_PREFLIGHT_OUTPUT_FORBIDDEN",
        )


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
