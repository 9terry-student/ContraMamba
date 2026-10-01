#!/usr/bin/env python3
"""Gen5 Stage E learned-B-plane positive-control runner.

Design + implementation authority:
    c9ef6c55448a3458b46f5b76b5c88244cc7b726e

Only --static-verify-only is authorized before a later execution authority and
an exact implementation-freeze commit.
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

from contramamba.gen5_phase2_state_update_ownership import (  # noqa: E402
    load_frozen_owner_bases,
    parent_parameter_fingerprint,
    phase2_final_three_way_ce,
)
from contramamba.gen5_stage_e_causal_plane_bottleneck import (  # noqa: E402
    EXPECTED_TRAINABLE_NUMEL,
    stage_e_optimizer_parameters,
    stage_e_parameter_audit,
)
from contramamba.gen5_stage_e_learned_b_plane_positive_control import (  # noqa: E402
    ARM,
    FIXED_PLANE_RESIDUAL_ATOL,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneCorrection,
    install_learned_b_plane_layer22_wrapper,
    learned_b_fixed_plane_geometry,
    load_seed_matched_learned_plane,
    tensor_sha256,
)
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402
from scripts import train_reason_router_gen5_stage_e_causal_plane_bottleneck as stagee  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "c9ef6c55448a3458b46f5b76b5c88244cc7b726e"
)
AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_learned_b_plane_positive_control_"
    "design_implementation_authority_spec_candidate.md"
)
AUTHORITY_BLOB = "987426f4eb19a66803c2715ee83d6c4a135a7518"

STAGE_E_EVIDENCE_FREEZE_COMMIT = (
    "382a4961ee701ddf38b39ef9fa75aa932ce34f39"
)
STAGE_E_IMPLEMENTATION_FREEZE_COMMIT = (
    "88625179cd63d4e61e8719045b9a58b611f9825e"
)
PHASE3A_SOURCE_EXECUTION_COMMIT = (
    "d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e"
)

FROZEN_BLOBS = {
    AUTHORITY_PATH: AUTHORITY_BLOB,
    "src/contramamba/gen5_phase2_state_update_ownership.py":
        "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "scripts/train_reason_router_gen5_phase3a_contention.py":
        "45e333128c4fa31f4f502c5fdcf324243c1084fd",
    "src/contramamba/gen5_stage_e_causal_plane_bottleneck.py":
        "8c67d97362da4aba94816cfaccac436210fed442",
    "scripts/train_reason_router_gen5_stage_e_causal_plane_bottleneck.py":
        "dbda5607c5edbe0b70e32209715b381dbaa33931",
    "tests/test_reason_router_gen5_stage_e_causal_plane_bottleneck.py":
        "983b89f1bdf0c013d051629e5ed63cbe6d750dbb",
}

IMPLEMENTATION_PATHS = frozenset({
    "src/contramamba/gen5_stage_e_learned_b_plane_positive_control.py",
    "scripts/train_reason_router_gen5_stage_e_learned_b_plane_positive_control.py",
    "tests/test_reason_router_gen5_stage_e_learned_b_plane_positive_control.py",
})

EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_learned_b_plane_positive_control_"
    "execution_authority_spec_candidate.md"
)
PREFLIGHT_PREFIX = (
    "reports/reason_router_gen5_stage_e_bfree_cuda_preflight_runs/"
)
RUN_PREFIX = (
    "reports/reason_router_gen5_stage_e_bfree_positive_control_runs/"
)

TRAIN_ROWS = stagee.TRAIN_ROWS
DEV_ROWS = stagee.DEV_ROWS
SPLIT_SEED = stagee.SPLIT_SEED
BACKBONE_STREAM_ROWS = stagee.BACKBONE_STREAM_ROWS
PRESSURE = "P0"
TOTAL_OPTIMIZER_STEPS = stagee.TOTAL_OPTIMIZER_STEPS
LEARNING_RATE = stagee.LEARNING_RATE
WEIGHT_DECAY = stagee.WEIGHT_DECAY
GRADIENT_CLIP_NORM = stagee.GRADIENT_CLIP_NORM
PARENT_CHECKPOINT_SHA256 = stagee.PARENT_CHECKPOINT_SHA256
ZERO_DEV_CE_P0 = stagee.ZERO_DEV_CE_P0
FREE_P0_GAIN_BY_SEED = dict(stagee.FREE_P0_GAIN_BY_SEED)

FROZEN_STAGE_E_RECOVERY = {
    6201: {"E-R22": 0.001975318561968087, "E-C22": 0.0017156096081790623},
    6202: {"E-R22": 0.002009419003162425, "E-C22": 0.0020609486561624537},
    6203: {"E-R22": 0.0024332976314431222, "E-C22": 0.0018053421563742791},
}
MATRIX_GPU_COUNT = 2
MATRIX_WORKER_SEEDS = ((6201, 6203), (6202,))


class PositiveControlRunnerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PositiveControlRunnerError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PositiveControlRunnerError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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
    require(
        branch in (EXPECTED_BRANCH, ""),
        f"BRANCH:{branch}",
    )
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
            STAGE_E_EVIDENCE_FREEZE_COMMIT,
            head,
        ) == 0,
        "STAGE_E_EVIDENCE_NOT_ANCESTOR",
    )

    status = _status_paths()
    if allow_opening_worktree:
        require(
            status <= IMPLEMENTATION_PATHS,
            f"UNAUTHORIZED_WORKTREE_PATHS:{sorted(status)}",
        )
    else:
        require(not status, f"WORKTREE_NOT_CLEAN:{sorted(status)}")

    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB:{path}:{observed}")

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
            require(frozen_blob == live_blob, f"IMPLEMENTATION_DRIFT:{path}")

    return {
        "branch": branch,
        "head": head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "implementation_freeze_commit": implementation_freeze_commit,
    }


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"OUTPUT_COLLISION:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stagee.canonical_json_bytes(dict(value)) + b"\n")


def _validate_frozen_contract() -> None:
    require(TRAINING_SEEDS == (6201, 6202, 6203), "TRAINING_SEEDS")
    require(ARM == "E-BFREE", "ARM")
    require(TRAIN_ROWS == 3360, "TRAIN_ROWS")
    require(DEV_ROWS == 840, "DEV_ROWS")
    require(SPLIT_SEED == 16384, "SPLIT_SEED")
    require(PRESSURE == "P0", "PRESSURE")
    require(TOTAL_OPTIMIZER_STEPS == 20, "OPTIMIZER_STEPS")
    require(LEARNING_RATE == 0.001, "LEARNING_RATE")
    require(WEIGHT_DECAY == 0.0001, "WEIGHT_DECAY")
    require(GRADIENT_CLIP_NORM == 5.0, "GRADIENT_CLIP_NORM")
    require(EXPECTED_TRAINABLE_NUMEL == 1540, "TRAINABLE_NUMEL")
    require(set(SOURCE_CORRECTIONS) == set(TRAINING_SEEDS), "SOURCE_SEEDS")


def _validate_execution_authority(args: argparse.Namespace) -> None:
    require(args.execution_authority_commit is not None, "EXECUTION_AUTHORITY_COMMIT_REQUIRED")
    require(args.implementation_freeze_commit is not None, "IMPLEMENTATION_FREEZE_COMMIT_REQUIRED")
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
    require(authority_blob == live_blob, "EXECUTION_AUTHORITY_DRIFT")
    text = authority.read_text(encoding="utf-8")
    required_tokens = (
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_BFREE_THREE_CELL_POSITIVE_CONTROL",
        f"IMPLEMENTATION_FREEZE_COMMIT={args.implementation_freeze_commit}",
        "CUDA_PREFLIGHT_ALLOWED=YES",
        "TRAINING_ALLOWED=YES_EXACT_STAGE_E_BFREE_THREE_CELL",
        "EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV",
        "BACKWARD_ALLOWED=YES_TRAINING_ONLY",
        "OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
    )
    for token in required_tokens:
        require(token in text, f"EXECUTION_AUTHORITY_TOKEN:{token}")


def _validate_seed_matched_initialization(seed: int, q: torch.Tensor) -> dict[str, Any]:
    r22, c22, _ = load_frozen_owner_bases(ROOT)
    reference = stagee.install_stage_e_layer22_wrapper  # prove frozen import remains available
    del reference
    from contramamba.gen5_stage_e_causal_plane_bottleneck import (
        FixedPlaneStateWriteCorrection,
    )
    stage_e_ref = FixedPlaneStateWriteCorrection(
        arm="E-R22",
        r22=r22,
        c22=c22,
        seed=seed,
    )
    candidate = LearnedBPlaneCorrection(q_bfree=q, seed=seed)
    same_a = torch.equal(
        stage_e_ref.A_theta.weight.detach().cpu(),
        candidate.A_theta.weight.detach().cpu(),
    )
    require(same_a, f"A_INIT_NOT_STAGE_E_MATCHED:{seed}")
    require(
        int(torch.count_nonzero(candidate.M_theta.weight).item()) == 0,
        f"M_INIT_NONZERO:{seed}",
    )
    trainable_numel = sum(
        p.numel() for p in candidate.parameters() if p.requires_grad
    )
    require(trainable_numel == EXPECTED_TRAINABLE_NUMEL, f"TRAINABLE_NUMEL:{seed}")
    return {
        "A_theta_sha256": tensor_sha256(candidate.A_theta.weight),
        "M_theta_sha256": tensor_sha256(candidate.M_theta.weight),
        "stage_e_A_init_exact_match": True,
        "trainable_numel": int(trainable_numel),
    }


def run_static_verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = authenticate_repo(
        args.expected_head,
        allow_opening_worktree=args.allow_opening_worktree,
    )
    _validate_frozen_contract()
    static = p3a.validate_static_artifacts()

    plane_rows: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        q, source_meta = load_seed_matched_learned_plane(ROOT, seed)
        init_meta = _validate_seed_matched_initialization(seed, q)
        plane_rows[str(seed)] = {
            "source": source_meta,
            "initialization": init_meta,
        }

    report = {
        "schema_version": "GEN5_STAGE_E_BFREE_STATIC_VERIFY_V1",
        "result": "PASS_GEN5_STAGE_E_BFREE_STATIC_VERIFY",
        "execution_head": args.expected_head,
        "repository": repo,
        "static_tree_sha256": static["tree"]["tree_sha256"],
        "train_order_sha256": p3a.row_order_sha256(static["train_rows"]),
        "dev_order_sha256": p3a.row_order_sha256(static["dev_rows"]),
        "seeds": list(TRAINING_SEEDS),
        "arm": ARM,
        "pressure": PRESSURE,
        "seed_matched_planes": plane_rows,
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

    print("GEN5_STAGE_E_BFREE_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"STATIC_TREE_SHA256={static['tree']['tree_sha256']}")
    print("ARM=E-BFREE")
    print("SEEDS=6201,6202,6203")
    print("PRESSURE=P0")
    print("EXPECTED_TRAINABLE_NUMEL=1540")
    for seed in TRAINING_SEEDS:
        source = plane_rows[str(seed)]["source"]
        init = plane_rows[str(seed)]["initialization"]
        print(
            f"SEED={seed} "
            f"SOURCE_SHA256={source['source_file_sha256']} "
            f"Q_SHA256={source['Q_tensor_sha256']} "
            f"B_RANK={source['source_B_rank']} "
            f"QR_RECON_REL={source['qr_reconstruction_relative']:.12g} "
            f"A_INIT_SHA256={init['A_theta_sha256']}"
        )
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


def _prepare_runtime_model(
    *,
    snapshot: Path,
    checkpoint_path: Path,
    seed: int,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    require(seed in TRAINING_SEEDS, f"SEED:{seed}")
    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership as p2train,
    )
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    kernels = kernel_compat.load_exact_fast_kernels()
    q, source_meta = load_seed_matched_learned_plane(ROOT, seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, constructor_counts = p2train._load_parent_model(
        snapshot=snapshot,
        checkpoint=checkpoint_path,
        device=torch.device("cuda:0"),
        kernel_compat=kernel_compat,
        kernels=kernels,
    )
    parent_before = parent_parameter_fingerprint(model)
    wrapper = install_learned_b_plane_layer22_wrapper(
        model,
        q_bfree=q,
        seed=seed,
    )
    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_INSTALL_MUTATION",
    )
    audit = stage_e_parameter_audit(model)
    require(audit["trainable_tensor_count"] == 2, "TRAINABLE_TENSOR_COUNT")
    require(audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL, "TRAINABLE_NUMEL")
    require(
        int(torch.count_nonzero(wrapper.correction.M_theta.weight).item()) == 0,
        "M_NOT_ZERO_INITIALIZED",
    )
    return model, wrapper, {
        "runtime": runtime,
        "constructor_counts": constructor_counts,
        "parent_before": parent_before,
        "parameter_audit": audit,
        "source_plane": source_meta,
    }


def run_cuda_preflight(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()
    require(args.seed in TRAINING_SEEDS, "PREFLIGHT_SEED")
    require(args.preflight_output is not None, "PREFLIGHT_OUTPUT_REQUIRED")

    static, encoded, snapshot, checkpoint_path = stagee._prepare_runtime_inputs(args)
    model, wrapper, runtime_meta = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    features, labels = stagee._bundle_to_device(
        encoded["train_bundle"], torch.device("cuda:0")
    )

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model.zero_grad(set_to_none=True)

    output, chunks = stagee._streamed_forward(model, features)
    logits = output["logits"]
    require(tuple(logits.shape) == (TRAIN_ROWS, 3), "PREFLIGHT_LOGIT_SHAPE")
    require(bool(torch.isfinite(logits).all().item()), "PREFLIGHT_LOGIT_FINITE")
    loss = phase2_final_three_way_ce(logits, labels)
    require(bool(torch.isfinite(loss).item()), "PREFLIGHT_LOSS")
    loss.backward()

    a_grad = wrapper.correction.A_theta.weight.grad
    m_grad = wrapper.correction.M_theta.weight.grad
    require(a_grad is not None and m_grad is not None, "PREFLIGHT_GRAD_MISSING")
    require(bool(torch.isfinite(a_grad).all().item()), "PREFLIGHT_A_GRAD_NONFINITE")
    require(bool(torch.isfinite(m_grad).all().item()), "PREFLIGHT_M_GRAD_NONFINITE")
    parent_grads = [
        name for name, parameter in model.named_parameters()
        if ".correction." not in name and parameter.grad is not None
    ]
    require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")

    geometry = learned_b_fixed_plane_geometry(wrapper.correction)
    report = {
        "schema_version": "GEN5_STAGE_E_BFREE_CUDA_PREFLIGHT_V1",
        "result": "PASS_GEN5_STAGE_E_BFREE_CUDA_PREFLIGHT",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "seed": args.seed,
        "arm": ARM,
        "pressure": PRESSURE,
        "source_plane": runtime_meta["source_plane"],
        "train_order_sha256": p3a.row_order_sha256(static["train_rows"]),
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "loss": float(loss.detach().cpu().item()),
        "geometry": geometry,
        "stream_chunk_count": chunks,
        "backward_executed": True,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "training_executed": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(Path(args.preflight_output), report)

    print("GEN5_STAGE_E_BFREE_CUDA_PREFLIGHT_PASS")
    print(f"SEED={args.seed}")
    print("ARM=E-BFREE")
    print(f"STREAM_CHUNKS={chunks}")
    print("BACKWARD_EXECUTED=True")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("TRAINING_EXECUTED=False")
    return report


def _checkpoint_payload(
    *,
    wrapper: Any,
    args: argparse.Namespace,
    source_meta: Mapping[str, Any],
) -> dict[str, Any]:
    a = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    m = wrapper.correction.M_theta.weight.detach().cpu().contiguous()
    return {
        "schema_version": "GEN5_STAGE_E_BFREE_FINAL_CORRECTION_V1",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "seed": args.seed,
        "arm": ARM,
        "pressure": PRESSURE,
        "source_plane": dict(source_meta),
        "state_dict": {
            "A_theta.weight": a,
            "M_theta.weight": m,
        },
        "tensor_sha256": {
            "A_theta.weight": tensor_sha256(a),
            "M_theta.weight": tensor_sha256(m),
        },
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()
    require(args.seed in TRAINING_SEEDS, f"SEED:{args.seed}")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")

    static, encoded, snapshot, checkpoint_path = stagee._prepare_runtime_inputs(args)
    cell_dir = Path(args.output_root) / f"seed{args.seed}" / ARM
    require(not cell_dir.exists(), f"CELL_OUTPUT_COLLISION:{cell_dir}")
    cell_dir.mkdir(parents=True, exist_ok=False)

    model, wrapper, runtime_meta = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    parent_before = runtime_meta["parent_before"]
    train_features, train_labels = stagee._bundle_to_device(
        encoded["train_bundle"], torch.device("cuda:0")
    )
    dev_features, dev_labels = stagee._bundle_to_device(
        encoded["dev_bundle"], torch.device("cuda:0")
    )

    optimizer_parameters = stage_e_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership as p2train,
    )
    step0_rng_state = p2train._capture_rng_state()
    losses: list[float] = []
    grad_norms: list[float] = []

    for step in range(TOTAL_OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        output, chunks = stagee._streamed_forward(model, train_features)
        require(
            chunks == math.ceil(TRAIN_ROWS / BACKBONE_STREAM_ROWS),
            f"CHUNKS:{chunks}",
        )
        logits = output["logits"]
        require(tuple(logits.shape) == (TRAIN_ROWS, 3), "TRAIN_LOGIT_SHAPE")
        require(bool(torch.isfinite(logits).all().item()), f"TRAIN_LOGIT_NONFINITE:{step}")
        loss = phase2_final_three_way_ce(logits, train_labels)
        require(bool(torch.isfinite(loss).item()), f"LOSS_NONFINITE:{step}")
        loss.backward()

        a_grad = wrapper.correction.A_theta.weight.grad
        m_grad = wrapper.correction.M_theta.weight.grad
        require(a_grad is not None and m_grad is not None, f"GRAD_MISSING:{step}")
        require(
            bool(torch.isfinite(a_grad).all().item())
            and bool(torch.isfinite(m_grad).all().item()),
            f"GRAD_NONFINITE:{step}",
        )
        parent_grads = [
            name for name, parameter in model.named_parameters()
            if ".correction." not in name and parameter.grad is not None
        ]
        require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")
        clipped = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters, GRADIENT_CLIP_NORM
        )
        require(bool(torch.isfinite(clipped).item()), "GRAD_NORM_NONFINITE")
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(float(clipped.detach().cpu().item()))
        del output, logits, loss
        torch.cuda.synchronize()

    require(parent_parameter_fingerprint(model) == parent_before, "PARENT_MUTATION_AFTER_TRAINING")
    post_training_rng_state = p2train._capture_rng_state()
    optimizer.zero_grad(set_to_none=True)
    p2train._restore_rng_state(step0_rng_state)
    post_output, post_chunks = stagee._streamed_forward(model, train_features)
    post_loss_tensor = phase2_final_three_way_ce(post_output["logits"], train_labels)
    require(bool(torch.isfinite(post_loss_tensor).item()), "POST_STEP20_LOSS_NONFINITE")
    post_step20_loss = float(post_loss_tensor.detach().cpu().item())
    del post_output, post_loss_tensor
    p2train._restore_rng_state(post_training_rng_state)
    p2train._require_rng_state_equal(
        post_training_rng_state,
        p2train._capture_rng_state(),
        "BFREE_POST_STEP20_DIAGNOSTIC_RESTORE",
    )
    require(parent_parameter_fingerprint(model) == parent_before, "PARENT_MUTATION_AFTER_POST_STEP20")

    model.eval()
    with torch.no_grad():
        dev_output, dev_chunks = stagee._streamed_forward(model, dev_features)
        dev_logits = dev_output["logits"]
        require(tuple(dev_logits.shape) == (DEV_ROWS, 3), "DEV_LOGIT_SHAPE")
        require(bool(torch.isfinite(dev_logits).all().item()), "DEV_LOGIT_NONFINITE")
        dev_ce_tensor = phase2_final_three_way_ce(dev_logits, dev_labels)
        dev_ce = float(dev_ce_tensor.detach().cpu().item())
        predictions = torch.argmax(dev_logits, dim=-1)
        dev_accuracy = float(
            (predictions == dev_labels).to(torch.float64).mean().cpu().item()
        )

    gain = ZERO_DEV_CE_P0 - dev_ce
    free_gain = FREE_P0_GAIN_BY_SEED[args.seed]
    recovery = gain / free_gain
    geometry = learned_b_fixed_plane_geometry(wrapper.correction)
    require(
        float(geometry["fixed_plane_residual_max_abs"]) <= FIXED_PLANE_RESIDUAL_ATOL,
        "FIXED_PLANE_RESIDUAL",
    )

    checkpoint_out = cell_dir / "final_correction.pt"
    torch.save(
        _checkpoint_payload(
            wrapper=wrapper,
            args=args,
            source_meta=runtime_meta["source_plane"],
        ),
        checkpoint_out,
    )
    audit = stage_e_parameter_audit(model)
    report = {
        "schema_version": "GEN5_STAGE_E_BFREE_TRAINING_REPORT_V1",
        "result": "PASS_GEN5_STAGE_E_BFREE_TRAINING_CELL",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "seed": args.seed,
        "arm": ARM,
        "selected_basis": "SEED_MATCHED_PHASE3A_P0_FINAL_B_SPAN",
        "pressure": PRESSURE,
        "source_plane": runtime_meta["source_plane"],
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "train_order_sha256": p3a.row_order_sha256(static["train_rows"]),
        "dev_order_sha256": p3a.row_order_sha256(static["dev_rows"]),
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "training_losses": losses,
        "step0_loss": losses[0],
        "last_preupdate_loss": losses[-1],
        "post_step20_matched_rng_loss": post_step20_loss,
        "loss_gate_rng_policy": "MATCH_STEP0_TRAIN_MODE_RNG",
        "gradient_norms_before_clip": grad_norms,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "dev_final_3way_ce": dev_ce,
        "dev_accuracy": dev_accuracy,
        "zero_dev_ce_p0": ZERO_DEV_CE_P0,
        "gain_vs_zero": gain,
        "frozen_free_p0_gain_seed_matched": free_gain,
        "recovery_vs_free_p0": recovery,
        "frozen_stage_e_recovery_seed_matched": FROZEN_STAGE_E_RECOVERY[args.seed],
        "fixed_plane_geometry": geometry,
        "trainable_tensor_names": audit["trainable_names"],
        "trainable_tensor_count": audit["trainable_tensor_count"],
        "trainable_numel": audit["trainable_numel"],
        "A_theta_sha256": tensor_sha256(wrapper.correction.A_theta.weight),
        "M_theta_sha256": tensor_sha256(wrapper.correction.M_theta.weight),
        "parent_signature_before": parent_before,
        "parent_signature_after": parent_parameter_fingerprint(model),
        "runtime": runtime_meta["runtime"],
        "train_stream_chunks": math.ceil(TRAIN_ROWS / BACKBONE_STREAM_ROWS),
        "post_step20_stream_chunks": post_chunks,
        "dev_stream_chunks": dev_chunks,
        "final_correction_file_sha256": stagee.sha256_file(checkpoint_out),
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "optimizer_step_count": TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(cell_dir / "training_report.json", report)
    provenance = {
        "schema_version": "GEN5_STAGE_E_BFREE_TRAINING_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "seed": args.seed,
        "arm": ARM,
        "pressure": PRESSURE,
        "Q_tensor_sha256": runtime_meta["source_plane"]["Q_tensor_sha256"],
        "training_report_sha256": stagee.sha256_file(cell_dir / "training_report.json"),
        "final_correction_file_sha256": stagee.sha256_file(checkpoint_out),
        "training_executed": True,
        "backward_executed": True,
        "optimizer_step_count": TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(cell_dir / "run_provenance.json", provenance)
    print(
        f"GEN5_STAGE_E_BFREE_CELL_PASS seed={args.seed} "
        f"dev_ce={dev_ce:.12g} gain={gain:.12g} recovery={recovery:.12g}"
    )
    return report


def _validate_two_t4s() -> dict[str, Any]:
    return stagee._validate_two_t4s()


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
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    gpu_meta = _validate_two_t4s()

    output_root = Path(args.output_root)
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")
    scratch_parent = Path(tempfile.mkdtemp(prefix="contramamba_stage_e_bfree_"))
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
                "import sys; "
                "from scripts."
                "train_reason_router_gen5_stage_e_learned_b_plane_positive_control "
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

    return_codes = [p.wait() for p in processes]
    if any(code != 0 for code in return_codes):
        output_root.mkdir(parents=True, exist_ok=False)
        fail_dir = output_root / "failed_worker_logs"
        fail_dir.mkdir()
        for src in worker_logs:
            if src.exists():
                shutil.copy2(src, fail_dir / src.name)
        raise PositiveControlRunnerError(
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
    summary = {
        "schema_version": "GEN5_STAGE_E_BFREE_MATRIX_SUMMARY_V1",
        "result": "PASS_GEN5_STAGE_E_BFREE_THREE_CELL_MATRIX",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "gpu_meta": gpu_meta,
        "worker_seeds": [list(x) for x in MATRIX_WORKER_SEEDS],
        "cell_count": 3,
        "cells": reports,
        "mean_recovery_E_BFREE": mean_recovery,
        "frozen_stage_e_mean_recovery_E_R22": 0.00213934506552454,
        "frozen_stage_e_mean_recovery_E_C22": 0.00186063347357193,
        "training_executed": True,
        "backward_executed": True,
        "optimizer_step_count": 3 * TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(output_root / "matrix_summary.json", summary)
    provenance = {
        "schema_version": "GEN5_STAGE_E_BFREE_MATRIX_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "matrix_summary_sha256": stagee.sha256_file(output_root / "matrix_summary.json"),
        "cell_count": 3,
        "optimizer_step_count": 3 * TOTAL_OPTIMIZER_STEPS,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(output_root / "run_provenance.json", provenance)
    shutil.rmtree(scratch_parent)

    print("GEN5_STAGE_E_BFREE_THREE_CELL_MATRIX_PASS")
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
    print(f"SUMMARY={output_root / 'matrix_summary.json'}")
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
        require(args.implementation_freeze_commit is None, "STATIC_IMPLEMENTATION_FREEZE_FORBIDDEN")
        require(args.execution_authority_commit is None, "STATIC_EXECUTION_AUTHORITY_FORBIDDEN")
        require(args.seed is None, "STATIC_SEED_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(args.preflight_output is None, "STATIC_PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.output_root is None, "STATIC_OUTPUT_ROOT_FORBIDDEN")
        require(args.model_snapshot is None, "STATIC_MODEL_SNAPSHOT_FORBIDDEN")
        require(args.tokenizer_snapshot is None, "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN")
        return

    require(not args.allow_opening_worktree, "RUNTIME_OPENING_WORKTREE_FORBIDDEN")
    require(args.implementation_freeze_commit is not None, "RUNTIME_IMPLEMENTATION_FREEZE_REQUIRED")
    require(args.execution_authority_commit is not None, "RUNTIME_EXECUTION_AUTHORITY_REQUIRED")
    require(args.checkpoint is not None, "RUNTIME_CHECKPOINT_REQUIRED")

    if args.cuda_preflight_only:
        require(args.seed in TRAINING_SEEDS, "PREFLIGHT_SEED_REQUIRED")
        require(args.preflight_output is not None, "PREFLIGHT_OUTPUT_REQUIRED")
        require(args.output_root is None, "PREFLIGHT_OUTPUT_ROOT_FORBIDDEN")
    elif args.run_cell:
        require(args.seed in TRAINING_SEEDS, "RUN_CELL_SEED_REQUIRED")
        require(args.output_root is not None, "RUN_CELL_OUTPUT_ROOT_REQUIRED")
        require(args.preflight_output is None, "RUN_CELL_PREFLIGHT_OUTPUT_FORBIDDEN")
    else:
        require(args.seed is None, "MATRIX_SEED_FORBIDDEN")
        require(args.output_root is not None, "MATRIX_OUTPUT_ROOT_REQUIRED")
        require(args.preflight_output is None, "MATRIX_PREFLIGHT_OUTPUT_FORBIDDEN")


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
