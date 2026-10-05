#!/usr/bin/env python3
"""Gen5 A-init × training-RNG causal intervention runner.

Original implementation authority:
    d0c86e8725e5df9def6acad323f3cce155b9daea
Runtime-hash recovery implementation authority:
    3f5d267508d3efaf123b93563f1563c20db62fe1

This implementation separates the seed that initializes A_theta from the seed
used for the remaining stochastic training path.

The recovery removes cross-PyTorch fixed raw-byte A-init hashes from the
scientific identity contract. Runtime A authentication is exact same-process
reconstruction. CUDA preflight / cell / matrix modes require a corrected
implementation freeze and a new recovery execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contramamba.gen5_phase2_state_update_ownership import (  # noqa: E402
    correction_optimizer_parameters,
    correction_parameter_audit,
    install_phase2_layer22_wrapper,
    load_frozen_owner_bases,
    parent_parameter_fingerprint,
    phase2_final_three_way_ce,
)
from contramamba.gen5_phase3_causal_role_contention import (  # noqa: E402
    contention_fractions,
    derive_strong_partition,
    load_frozen_planes,
)
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "d0c86e8725e5df9def6acad323f3cce155b9daea"
)
MECHANISM_FREEZE_COMMIT = (
    "155c1898f9f2d6dd0765fc93cc156190d8a08708"
)
FAILED_EXECUTION_AUTHORITY_COMMIT = (
    "54a0bffa075ac7c2f25147ff53fb971412b99749"
)
RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT = (
    "3f5d267508d3efaf123b93563f1563c20db62fe1"
)

AUTHORITY_PATH = (
    "reports/reason_router_gen5_ainit_rng_causal_intervention_"
    "design_implementation_authority_spec_candidate.md"
)
MECHANISM_REPORT_PATH = (
    "reports/reason_router_gen5_cross_seed_initialization_anchoring_"
    "mechanism_report_candidate.md"
)
FAILED_EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_ainit_rng_causal_intervention_"
    "execution_authority_spec_candidate.md"
)
RECOVERY_IMPLEMENTATION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_ainit_rng_runtime_hash_recovery_"
    "implementation_authority_spec_candidate.md"
)
EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_ainit_rng_runtime_hash_recovery_"
    "execution_authority_spec_candidate.md"
)

IMPLEMENTATION_PATHS = frozenset({
    "scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py",
    "tests/test_reason_router_gen5_ainit_rng_causal_intervention.py",
})

PREFLIGHT_PREFIX = (
    "reports/reason_router_gen5_ainit_rng_causal_intervention_cuda_preflight_runs/"
)
RUN_PREFIX = (
    "reports/reason_router_gen5_ainit_rng_causal_intervention_runs/"
)

FROZEN_BLOBS = {
    AUTHORITY_PATH: "4bb4c837c760ca17659e40f9154e722697ef087c",
    MECHANISM_REPORT_PATH: "cd0e0c9a4d64a98e5b50108d1c8f17fc58f4bd95",
    FAILED_EXECUTION_AUTHORITY_PATH:
        "7ae466b4b1cba3520e5bf9b180eb45959560d6a4",
    RECOVERY_IMPLEMENTATION_AUTHORITY_PATH:
        "d87721afbbb6c6a396c04a167d29234abfe3f640",
    "scripts/train_reason_router_gen5_phase3a_contention.py":
        "45e333128c4fa31f4f502c5fdcf324243c1084fd",
    "src/contramamba/gen5_phase2_state_update_ownership.py":
        "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "src/contramamba/gen5_phase3_causal_role_contention.py":
        "d8cd9873df23c7cb31f1bbb3c981672295b1af7a",
}

FACTOR_SEEDS = (6201, 6202, 6203)
PRESSURE = "P0"
ARM = "G5-C0"

OFFDIAGONAL_CELLS = (
    (6201, 6202),
    (6201, 6203),
    (6202, 6201),
    (6202, 6203),
    (6203, 6201),
    (6203, 6202),
)

MATRIX_GPU_COUNT = 2
MATRIX_WORKER_CELLS = (
    ((6201, 6202), (6202, 6203), (6203, 6201)),
    ((6201, 6203), (6202, 6201), (6203, 6202)),
)
GPU_TOPOLOGY = "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"


DIAGONAL_SOURCES = {
    6201: (
        "reports/reason_router_gen5_phase3a_training_runs/"
        "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
        "cells/seed6201/P0/final_correction.pt",
        "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
    ),
    6202: (
        "reports/reason_router_gen5_phase3a_training_runs/"
        "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
        "cells/seed6202/P0/final_correction.pt",
        "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
    ),
    6203: (
        "reports/reason_router_gen5_phase3a_training_runs/"
        "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
        "cells/seed6203/P0/final_correction.pt",
        "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
    ),
}

TRAIN_ROWS = p3a.TRAIN_ROWS
DEV_ROWS = p3a.DEV_ROWS
SPLIT_SEED = p3a.SPLIT_SEED
BACKBONE_STREAM_ROWS = p3a.BACKBONE_STREAM_ROWS
TOTAL_OPTIMIZER_STEPS = p3a.TOTAL_OPTIMIZER_STEPS
LEARNING_RATE = p3a.LEARNING_RATE
WEIGHT_DECAY = p3a.WEIGHT_DECAY
GRADIENT_CLIP_NORM = p3a.GRADIENT_CLIP_NORM
PARENT_CHECKPOINT_SHA256 = p3a.PARENT_CHECKPOINT_SHA256
STATIC_TREE_SHA256 = p3a.STATIC_TREE_SHA256
CONFIRMATORY_9601_9900_ALLOWED = False


class AInitRNGCausalInterventionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AInitRNGCausalInterventionError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AInitRNGCausalInterventionError(
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
    paths: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        require(len(line) >= 4, f"MALFORMED_STATUS:{line!r}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def _post_authority_path_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        normalized in IMPLEMENTATION_PATHS
        or normalized == FAILED_EXECUTION_AUTHORITY_PATH
        or normalized == RECOVERY_IMPLEMENTATION_AUTHORITY_PATH
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

    for ancestor, label in (
        (MECHANISM_FREEZE_COMMIT, "MECHANISM_FREEZE"),
        (IMPLEMENTATION_AUTHORITY_COMMIT, "IMPLEMENTATION_AUTHORITY"),
        (
            RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT,
            "RECOVERY_IMPLEMENTATION_AUTHORITY",
        ),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, head) == 0,
            f"{label}_NOT_ANCESTOR",
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
            require(
                frozen_blob == live_blob,
                f"IMPLEMENTATION_DRIFT:{path}",
            )

    return {
        "branch": branch,
        "head": head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "recovery_implementation_authority_commit":
            RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT,
        "failed_execution_authority_commit": FAILED_EXECUTION_AUTHORITY_COMMIT,
        "mechanism_freeze_commit": MECHANISM_FREEZE_COMMIT,
        "implementation_freeze_commit": implementation_freeze_commit,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"OUTPUT_COLLISION:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def reconstruct_a_init(a_init_seed: int) -> torch.Tensor:
    require(a_init_seed in FACTOR_SEEDS, f"A_INIT_SEED:{a_init_seed}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(a_init_seed)
    value = torch.empty((2, 768), dtype=torch.float32, device="cpu")
    nn.init.kaiming_uniform_(
        value,
        a=math.sqrt(5),
        generator=generator,
    )
    return value.contiguous()


def authenticate_live_a_init(
    live_weight: torch.Tensor,
    a_init_seed: int,
) -> dict[str, Any]:
    expected = reconstruct_a_init(a_init_seed)
    live = live_weight.detach().cpu().contiguous()
    expected_live = expected.to(dtype=live.dtype).contiguous()

    require(
        tuple(live.shape) == tuple(expected_live.shape) == (2, 768),
        f"A_INIT_SHAPE:{a_init_seed}:{tuple(live.shape)}",
    )
    require(
        torch.equal(live, expected_live),
        f"A_INIT_IDENTITY:{a_init_seed}:{tensor_sha256(live)}",
    )

    return {
        "a_init_sha256": tensor_sha256(live),
        "a_init_torch_version": str(torch.__version__),
        "a_init_authentication":
            "SAME_PROCESS_EXACT_TENSOR_IDENTITY_AFTER_LIVE_DTYPE_NORMALIZATION",
        "cross_version_fixed_hash_authority": False,
    }


def zero_b_init() -> torch.Tensor:
    return torch.zeros((24576, 2), dtype=torch.float32, device="cpu")


def cell_name(a_init_seed: int, training_rng_seed: int) -> str:
    return f"A{a_init_seed}-R{training_rng_seed}"


def validate_factor_cell(
    a_init_seed: int,
    training_rng_seed: int,
) -> None:
    require(a_init_seed in FACTOR_SEEDS, f"A_INIT_SEED:{a_init_seed}")
    require(
        training_rng_seed in FACTOR_SEEDS,
        f"TRAINING_RNG_SEED:{training_rng_seed}",
    )


def validate_offdiagonal_cell(
    a_init_seed: int,
    training_rng_seed: int,
) -> None:
    validate_factor_cell(a_init_seed, training_rng_seed)
    require(
        (a_init_seed, training_rng_seed) in OFFDIAGONAL_CELLS,
        f"NOT_AUTHORIZED_OFFDIAGONAL:{a_init_seed}:{training_rng_seed}",
    )


def _validate_contract() -> None:
    require(FACTOR_SEEDS == (6201, 6202, 6203), "FACTOR_SEEDS")
    require(PRESSURE == "P0", "PRESSURE")
    require(ARM == "G5-C0", "ARM")
    require(len(OFFDIAGONAL_CELLS) == 6, "OFFDIAGONAL_COUNT")
    require(
        all(a != r for a, r in OFFDIAGONAL_CELLS),
        "DIAGONAL_SCHEDULED",
    )
    require(
        set(OFFDIAGONAL_CELLS)
        == {
            (a, r)
            for a in FACTOR_SEEDS
            for r in FACTOR_SEEDS
            if a != r
        },
        "OFFDIAGONAL_SET",
    )

    require(MATRIX_GPU_COUNT == 2, "GPU_COUNT")
    require(len(MATRIX_WORKER_CELLS) == 2, "WORKER_COUNT")
    flattened = [
        cell
        for worker in MATRIX_WORKER_CELLS
        for cell in worker
    ]
    require(
        set(flattened) == set(OFFDIAGONAL_CELLS),
        "WORKER_CELL_SET",
    )
    require(len(set(flattened)) == 6, "WORKER_CELL_DUPLICATE")
    for worker in MATRIX_WORKER_CELLS:
        require(len(worker) == 3, "WORKER_CELL_COUNT")
        require(
            {a for a, _ in worker} == set(FACTOR_SEEDS),
            "WORKER_A_INIT_BALANCE",
        )
        require(
            {r for _, r in worker} == set(FACTOR_SEEDS),
            "WORKER_RNG_BALANCE",
        )

    require(TRAIN_ROWS == 3360, "TRAIN_ROWS")
    require(DEV_ROWS == 840, "DEV_ROWS")
    require(SPLIT_SEED == 16384, "SPLIT_SEED")
    require(TOTAL_OPTIMIZER_STEPS == 20, "OPTIMIZER_STEPS")
    require(LEARNING_RATE == 0.001, "LEARNING_RATE")
    require(WEIGHT_DECAY == 0.0001, "WEIGHT_DECAY")
    require(GRADIENT_CLIP_NORM == 5.0, "GRADIENT_CLIP_NORM")
    require(not CONFIRMATORY_9601_9900_ALLOWED, "CONFIRMATORY_ALLOWED")


def _validate_diagonal_sources() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for seed in FACTOR_SEEDS:
        relative_path, expected_sha = DIAGONAL_SOURCES[seed]
        path = ROOT / relative_path
        require(path.is_file(), f"DIAGONAL_SOURCE_MISSING:{seed}")
        observed_sha = sha256_file(path)
        require(
            observed_sha == expected_sha,
            f"DIAGONAL_SOURCE_SHA:{seed}:{observed_sha}",
        )
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        require(isinstance(payload, dict), f"DIAGONAL_PAYLOAD:{seed}")
        require(
            payload.get("schema_version")
            == "GEN5_PHASE3A_FINAL_CORRECTION_V1",
            f"DIAGONAL_SCHEMA:{seed}",
        )
        require(int(payload.get("seed", -1)) == seed, f"DIAGONAL_SEED:{seed}")
        require(payload.get("arm") == ARM, f"DIAGONAL_ARM:{seed}")
        require(
            payload.get("pressure") == PRESSURE,
            f"DIAGONAL_PRESSURE:{seed}",
        )
        rows[str(seed)] = {
            "relative_path": relative_path,
            "file_sha256": observed_sha,
        }
    return rows


def _validate_execution_authority(args: argparse.Namespace) -> None:
    require(
        args.implementation_freeze_commit is not None,
        "IMPLEMENTATION_FREEZE_COMMIT_REQUIRED",
    )
    require(
        args.execution_authority_commit is not None,
        "EXECUTION_AUTHORITY_COMMIT_REQUIRED",
    )
    require(
        args.execution_authority_commit != FAILED_EXECUTION_AUTHORITY_COMMIT,
        "FAILED_EXECUTION_AUTHORITY_REUSE_FORBIDDEN",
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
    require(authority_blob == live_blob, "EXECUTION_AUTHORITY_DRIFT")

    text = authority.read_text(encoding="utf-8")
    required_tokens = (
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_GEN5_AINIT_RNG_CAUSAL_SIX_OFFDIAGONAL_MATRIX",
        f"IMPLEMENTATION_FREEZE_COMMIT={args.implementation_freeze_commit}",
        (
            "RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT="
            f"{RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT}"
        ),
        (
            "FAILED_EXECUTION_AUTHORITY_COMMIT="
            f"{FAILED_EXECUTION_AUTHORITY_COMMIT}"
        ),
        "RUNTIME_A_INIT_AUTHENTICATION=SAME_PROCESS_EXACT_TENSOR_IDENTITY",
        "CROSS_VERSION_A_INIT_HASH_AUTHORITY=NO",
        "CUDA_PREFLIGHT_ALLOWED=YES_OPTIONAL_OFFDIAGONAL",
        "TRAINING_ALLOWED=YES_EXACT_GEN5_AINIT_RNG_SIX_OFFDIAGONAL",
        "EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV",
        "BACKWARD_ALLOWED=YES_TRAINING_ONLY",
        "OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        f"GPU_TOPOLOGY={GPU_TOPOLOGY}",
    )
    for token in required_tokens:
        require(token in text, f"EXECUTION_AUTHORITY_TOKEN:{token}")


def run_static_verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = authenticate_repo(
        args.expected_head,
        allow_opening_worktree=args.allow_opening_worktree,
    )
    _validate_contract()
    static = p3a.validate_static_artifacts()
    diagonal = _validate_diagonal_sources()

    init_rows: dict[str, Any] = {}
    hashes = []
    tensors = []
    for seed in FACTOR_SEEDS:
        first = reconstruct_a_init(seed)
        second = reconstruct_a_init(seed)
        require(
            torch.equal(first, second),
            f"A_INIT_SAME_PROCESS_NONDETERMINISTIC:{seed}",
        )
        require(tuple(first.shape) == (2, 768), f"A_INIT_SHAPE:{seed}")
        require(first.dtype == torch.float32, f"A_INIT_DTYPE:{seed}")
        observed_hash = tensor_sha256(first)
        hashes.append(observed_hash)
        tensors.append(first)
        init_rows[str(seed)] = {
            "a_init_sha256_local_observed": observed_hash,
            "a_init_sha256_role":
                "NON_AUTHORITATIVE_RUNTIME_DIAGNOSTIC",
            "torch_version": str(torch.__version__),
            "shape": list(first.shape),
            "dtype": str(first.dtype),
        }
    require(len(set(hashes)) == 3, "A_INIT_HASHES_NOT_DISTINCT")
    require(
        all(
            not torch.equal(tensors[i], tensors[j])
            for i in range(len(tensors))
            for j in range(i + 1, len(tensors))
        ),
        "A_INIT_TENSORS_NOT_DISTINCT",
    )
    require(
        int(torch.count_nonzero(zero_b_init()).item()) == 0,
        "B_INIT_NONZERO",
    )

    report = {
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_STATIC_VERIFY_V1",
        "result": "PASS_GEN5_AINIT_RNG_CAUSAL_INTERVENTION_STATIC_VERIFY",
        "execution_head": args.expected_head,
        "repository": repo,
        "static_tree_sha256": static["tree"]["tree_sha256"],
        "factor_seeds": list(FACTOR_SEEDS),
        "pressure": PRESSURE,
        "arm": ARM,
        "offdiagonal_cells": [
            {
                "a_init_seed": a,
                "training_rng_seed": r,
                "cell_name": cell_name(a, r),
            }
            for a, r in OFFDIAGONAL_CELLS
        ],
        "worker_cells": [
            [
                {
                    "a_init_seed": a,
                    "training_rng_seed": r,
                    "cell_name": cell_name(a, r),
                }
                for a, r in worker
            ]
            for worker in MATRIX_WORKER_CELLS
        ],
        "gpu_count_for_future_execution": MATRIX_GPU_COUNT,
        "gpu_topology_for_future_execution": GPU_TOPOLOGY,
        "a_initializations": init_rows,
        "a_init_authentication_contract":
            "SAME_PROCESS_EXACT_TENSOR_IDENTITY",
        "cross_version_fixed_a_init_hash_authority": False,
        "static_torch_version": str(torch.__version__),
        "diagonal_sources": diagonal,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "checkpoint_loaded": False,
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

    print("GEN5_AINIT_RNG_CAUSAL_INTERVENTION_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print("FACTOR_SEEDS=6201,6202,6203")
    print("PRESSURE=P0")
    print("OFFDIAGONAL_CELLS=6")
    print("DIAGONAL_CELLS_REUSED=3")
    print("GPU_COUNT_FOR_FUTURE_EXECUTION=2")
    print(f"GPU_TOPOLOGY_FOR_FUTURE_EXECUTION={GPU_TOPOLOGY}")
    print(
        "WORKER0="
        + ",".join(cell_name(*cell) for cell in MATRIX_WORKER_CELLS[0])
    )
    print(
        "WORKER1="
        + ",".join(cell_name(*cell) for cell in MATRIX_WORKER_CELLS[1])
    )
    for seed in FACTOR_SEEDS:
        print(
            f"A_INIT_SEED={seed} "
            f"LOCAL_A_INIT_SHA256="
            f"{init_rows[str(seed)]['a_init_sha256_local_observed']} "
            "HASH_ROLE=NON_AUTHORITATIVE_RUNTIME_DIAGNOSTIC"
        )
    print("A_INIT_AUTHENTICATION=SAME_PROCESS_EXACT_TENSOR_IDENTITY")
    print("CROSS_VERSION_FIXED_A_INIT_HASH_AUTHORITY=False")
    print(f"STATIC_TORCH_VERSION={torch.__version__}")
    print("B_INIT_EXACT_ZERO=True")
    print("TRAINABLE_NUMEL_PER_CELL=50688")
    print("OPTIMIZER=torch.optim.AdamW")
    print(f"LEARNING_RATE={LEARNING_RATE}")
    print(f"WEIGHT_DECAY={WEIGHT_DECAY}")
    print(f"GRADIENT_CLIP_NORM={GRADIENT_CLIP_NORM}")
    print(f"OPTIMIZER_STEPS={TOTAL_OPTIMIZER_STEPS}")
    print("CHECKPOINT_LOADED=False")
    print("PARENT_MODEL_INSTANTIATED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("TRAINING_EXECUTED=False")
    print("TASK_EVALUATION_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")
    return report


def _prepare_runtime_model(
    *,
    snapshot: Path,
    checkpoint_path: Path,
    a_init_seed: int,
    training_rng_seed: int,
) -> tuple[
    torch.nn.Module,
    Any,
    dict[str, Any],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership as p2train,
    )

    # Low-level runtime preparation permits the full frozen 3x3 factor grid.
    # Historical public modes still call validate_offdiagonal_cell() before
    # reaching this helper, so their six-offdiagonal contract is unchanged.
    validate_factor_cell(a_init_seed, training_rng_seed)
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    kernels = kernel_compat.load_exact_fast_kernels()
    r22, c22, _basis_geometry = load_frozen_owner_bases(ROOT)

    # Parent construction/runtime stochasticity follows training_rng_seed.
    torch.manual_seed(training_rng_seed)
    torch.cuda.manual_seed_all(training_rng_seed)
    model, constructor_counts = p2train._load_parent_model(
        snapshot=snapshot,
        checkpoint=checkpoint_path,
        device=torch.device("cuda:0"),
        kernel_compat=kernel_compat,
        kernels=kernels,
    )

    parent_before = parent_parameter_fingerprint(model)

    # Correction A initialization follows only a_init_seed.
    wrapper = install_phase2_layer22_wrapper(
        model,
        arm=ARM,
        r22=r22,
        c22=c22,
        seed=a_init_seed,
    )
    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_INSTALL_MUTATION",
    )

    a_init_meta = authenticate_live_a_init(
        wrapper.correction.A_theta.weight,
        a_init_seed,
    )
    observed_a_sha = a_init_meta["a_init_sha256"]
    require(
        int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0,
        "B_NOT_ZERO_INITIALIZED",
    )

    mixer17 = model.mamba.layers[17].mixer
    partition = derive_strong_partition(
        mixer17.conv1d.weight,
        require_frozen_identity=True,
    )
    planes, plane_geometry = load_frozen_planes(ROOT)

    meta = {
        "runtime": runtime,
        "constructor_counts": constructor_counts,
        "parent_before": parent_before,
        "a_init_seed": a_init_seed,
        "training_rng_seed": training_rng_seed,
        "a_init_sha256": observed_a_sha,
        "a_init_torch_version": a_init_meta["a_init_torch_version"],
        "a_init_authentication": a_init_meta["a_init_authentication"],
        "cross_version_fixed_hash_authority":
            a_init_meta["cross_version_fixed_hash_authority"],
        "partition": {
            "mu_k2": partition["mu_k2"],
            "strong_count": int(partition["strong"].numel()),
            "strong_index_sha256": partition["strong_index_sha256"],
        },
        "plane_geometry": plane_geometry,
    }
    return model, wrapper, meta, partition["strong_mask"], planes


def _prepare_runtime_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership as p2train,
    )

    static = p3a.validate_static_artifacts()
    snapshot = p2train.resolve_exact_snapshot(args.model_snapshot)
    checkpoint_path = Path(args.checkpoint)
    p3a.validate_checkpoint(checkpoint_path)
    encoded = p3a.load_runtime_encoding(
        static,
        args.tokenizer_snapshot,
    )
    return static, encoded, snapshot, checkpoint_path


def _checkpoint_payload(
    *,
    wrapper: Any,
    args: argparse.Namespace,
    runtime_meta: Mapping[str, Any],
) -> dict[str, Any]:
    a = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    b = wrapper.correction.B_theta.weight.detach().cpu().contiguous()
    return {
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "recovery_implementation_authority_commit":
            RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT,
        "failed_execution_authority_commit": FAILED_EXECUTION_AUTHORITY_COMMIT,
        "mechanism_freeze_commit": MECHANISM_FREEZE_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "a_init_seed": args.a_init_seed,
        "training_rng_seed": args.training_rng_seed,
        "cell_name": cell_name(args.a_init_seed, args.training_rng_seed),
        "arm": ARM,
        "pressure": PRESSURE,
        "a_init_sha256": runtime_meta["a_init_sha256"],
        "a_init_torch_version": runtime_meta["a_init_torch_version"],
        "a_init_authentication": runtime_meta["a_init_authentication"],
        "cross_version_fixed_hash_authority":
            runtime_meta["cross_version_fixed_hash_authority"],
        "state_dict": {
            "A_theta.weight": a,
            "B_theta.weight": b,
        },
        "tensor_sha256": {
            "A_theta.weight": tensor_sha256(a),
            "B_theta.weight": tensor_sha256(b),
        },
    }


def run_cuda_preflight(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_contract()
    validate_offdiagonal_cell(
        args.a_init_seed,
        args.training_rng_seed,
    )
    require(args.preflight_output is not None, "PREFLIGHT_OUTPUT_REQUIRED")

    static, encoded, snapshot, checkpoint_path = _prepare_runtime_inputs(args)
    model, wrapper, runtime_meta, strong_mask, planes = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        a_init_seed=args.a_init_seed,
        training_rng_seed=args.training_rng_seed,
    )

    features, labels, active, targets = p3a._feature_batch_to_device(
        encoded["train_bundle"],
        torch.device("cuda:0"),
    )

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(args.training_rng_seed)
    torch.cuda.manual_seed_all(args.training_rng_seed)
    model.zero_grad(set_to_none=True)

    output, chunks = p3a._streamed_forward(
        model,
        features,
        active,
        targets,
        pressure=PRESSURE,
        strong_mask=strong_mask,
        planes=planes,
    )
    logits = output["logits"]
    require(tuple(logits.shape) == (TRAIN_ROWS, 3), "PREFLIGHT_LOGIT_SHAPE")
    require(bool(torch.isfinite(logits).all().item()), "PREFLIGHT_LOGIT_FINITE")
    loss = phase2_final_three_way_ce(logits, labels)
    require(bool(torch.isfinite(loss).item()), "PREFLIGHT_LOSS_FINITE")
    loss.backward()

    a_grad = wrapper.correction.A_theta.weight.grad
    b_grad = wrapper.correction.B_theta.weight.grad
    require(a_grad is not None and b_grad is not None, "PREFLIGHT_GRAD_MISSING")
    require(bool(torch.isfinite(a_grad).all().item()), "PREFLIGHT_A_GRAD_NONFINITE")
    require(bool(torch.isfinite(b_grad).all().item()), "PREFLIGHT_B_GRAD_NONFINITE")
    parent_grads = [
        name
        for name, parameter in model.named_parameters()
        if ".correction." not in name and parameter.grad is not None
    ]
    require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")

    report = {
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_CUDA_PREFLIGHT_V1",
        "result": "PASS_GEN5_AINIT_RNG_CAUSAL_CUDA_PREFLIGHT",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "a_init_seed": args.a_init_seed,
        "training_rng_seed": args.training_rng_seed,
        "cell_name": cell_name(args.a_init_seed, args.training_rng_seed),
        "pressure": PRESSURE,
        "a_init_sha256": runtime_meta["a_init_sha256"],
        "a_init_torch_version": runtime_meta["a_init_torch_version"],
        "a_init_authentication": runtime_meta["a_init_authentication"],
        "cross_version_fixed_hash_authority":
            runtime_meta["cross_version_fixed_hash_authority"],
        "train_order_sha256": p3a.row_order_sha256(static["train_rows"]),
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "loss": float(loss.detach().cpu().item()),
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

    print("GEN5_AINIT_RNG_CAUSAL_CUDA_PREFLIGHT_PASS")
    print(f"CELL={report['cell_name']}")
    print(f"A_INIT_SEED={args.a_init_seed}")
    print(f"TRAINING_RNG_SEED={args.training_rng_seed}")
    print(f"A_INIT_SHA256={runtime_meta['a_init_sha256']}")
    print(f"A_INIT_TORCH_VERSION={runtime_meta['a_init_torch_version']}")
    print(f"A_INIT_AUTHENTICATION={runtime_meta['a_init_authentication']}")
    print("CROSS_VERSION_FIXED_A_INIT_HASH_AUTHORITY=False")
    print("BACKWARD_EXECUTED=True")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("TRAINING_EXECUTED=False")
    print("TASK_EVALUATION_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    return report


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_contract()
    validate_offdiagonal_cell(
        args.a_init_seed,
        args.training_rng_seed,
    )
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")

    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership as p2train,
    )

    static, encoded, snapshot, checkpoint_path = _prepare_runtime_inputs(args)

    cell = cell_name(args.a_init_seed, args.training_rng_seed)
    cell_dir = Path(args.output_root) / cell
    require(not cell_dir.exists(), f"CELL_OUTPUT_COLLISION:{cell_dir}")
    cell_dir.mkdir(parents=True, exist_ok=False)

    model, wrapper, runtime_meta, strong_mask, planes = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        a_init_seed=args.a_init_seed,
        training_rng_seed=args.training_rng_seed,
    )
    parent_before = runtime_meta["parent_before"]

    train_features, train_labels, train_active, train_targets = (
        p3a._feature_batch_to_device(
            encoded["train_bundle"],
            torch.device("cuda:0"),
        )
    )
    dev_features, dev_labels, dev_active, dev_targets = (
        p3a._feature_batch_to_device(
            encoded["dev_bundle"],
            torch.device("cuda:0"),
        )
    )

    optimizer_parameters = correction_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    model.mamba.config.use_cache = False

    # Training stochasticity follows only training_rng_seed.
    torch.manual_seed(args.training_rng_seed)
    torch.cuda.manual_seed_all(args.training_rng_seed)
    step0_rng_state = p2train._capture_rng_state()

    losses: list[float] = []
    grad_norms: list[float] = []

    for step in range(TOTAL_OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        output, chunks = p3a._streamed_forward(
            model,
            train_features,
            train_active,
            train_targets,
            pressure=PRESSURE,
            strong_mask=strong_mask,
            planes=planes,
        )
        require(
            chunks == TRAIN_ROWS // BACKBONE_STREAM_ROWS,
            f"CHUNKS:{chunks}",
        )
        logits = output["logits"]
        require(tuple(logits.shape) == (TRAIN_ROWS, 3), "TRAIN_LOGIT_SHAPE")
        require(
            bool(torch.isfinite(logits).all().item()),
            f"TRAIN_LOGIT_NONFINITE:{step}",
        )
        loss = phase2_final_three_way_ce(logits, train_labels)
        require(bool(torch.isfinite(loss).item()), f"LOSS_NONFINITE:{step}")
        loss.backward()

        a_grad = wrapper.correction.A_theta.weight.grad
        b_grad = wrapper.correction.B_theta.weight.grad
        require(a_grad is not None and b_grad is not None, f"GRAD_MISSING:{step}")
        require(bool(torch.isfinite(a_grad).all().item()), f"A_GRAD_NONFINITE:{step}")
        require(bool(torch.isfinite(b_grad).all().item()), f"B_GRAD_NONFINITE:{step}")
        parent_grads = [
            name
            for name, parameter in model.named_parameters()
            if ".correction." not in name and parameter.grad is not None
        ]
        require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")

        clipped = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters,
            GRADIENT_CLIP_NORM,
        )
        require(bool(torch.isfinite(clipped).item()), "GRAD_NORM_NONFINITE")
        optimizer.step()
        require(
            bool(torch.isfinite(wrapper.correction.A_theta.weight).all().item())
            and bool(torch.isfinite(wrapper.correction.B_theta.weight).all().item()),
            "CORRECTION_PARAMETER_NONFINITE",
        )

        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(float(clipped.detach().cpu().item()))
        del output, logits, loss
        torch.cuda.synchronize()

    require(len(losses) == TOTAL_OPTIMIZER_STEPS, "LOSS_COUNT")
    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_MUTATION",
    )

    # Replay final train objective under exact step-0 stochastic RNG.
    post_training_rng_state = p2train._capture_rng_state()
    optimizer.zero_grad(set_to_none=True)
    p2train._restore_rng_state(step0_rng_state)

    post_output, post_chunks = p3a._streamed_forward(
        model,
        train_features,
        train_active,
        train_targets,
        pressure=PRESSURE,
        strong_mask=strong_mask,
        planes=planes,
    )
    require(
        post_chunks == TRAIN_ROWS // BACKBONE_STREAM_ROWS,
        f"POST_STEP20_CHUNKS:{post_chunks}",
    )
    post_logits = post_output["logits"]
    post_loss_tensor = phase2_final_three_way_ce(
        post_logits,
        train_labels,
    )
    require(
        bool(torch.isfinite(post_loss_tensor).item()),
        "POST_STEP20_LOSS_NONFINITE",
    )
    post_step20_loss = float(
        post_loss_tensor.detach().cpu().item()
    )
    del post_output, post_logits, post_loss_tensor

    p2train._restore_rng_state(post_training_rng_state)
    p2train._require_rng_state_equal(
        post_training_rng_state,
        p2train._capture_rng_state(),
        "POST_STEP20_DIAGNOSTIC_RESTORE",
    )

    # Frozen Phase3A dev evaluation: secondary to geometry endpoints.
    model.eval()
    with torch.no_grad():
        dev_output, dev_chunks = p3a._streamed_forward(
            model,
            dev_features,
            dev_active,
            dev_targets,
            pressure=PRESSURE,
            strong_mask=strong_mask,
            planes=planes,
        )
        dev_logits = dev_output["logits"]
        require(tuple(dev_logits.shape) == (DEV_ROWS, 3), "DEV_LOGIT_SHAPE")
        require(
            bool(torch.isfinite(dev_logits).all().item()),
            "DEV_LOGIT_NONFINITE",
        )
        dev_ce_tensor = phase2_final_three_way_ce(
            dev_logits,
            dev_labels,
        )
        dev_ce = float(dev_ce_tensor.detach().cpu().item())
        predictions = torch.argmax(dev_logits, dim=-1)
        dev_accuracy = float(
            (predictions == dev_labels)
            .to(torch.float64)
            .mean()
            .cpu()
            .item()
        )

    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_MUTATION_AFTER_EVAL",
    )

    r22, c22, _ = load_frozen_owner_bases(ROOT)
    geometry = contention_fractions(
        wrapper.correction.A_theta.weight,
        wrapper.correction.B_theta.weight,
        r22,
        c22,
    )

    checkpoint_out = cell_dir / "final_correction.pt"
    torch.save(
        _checkpoint_payload(
            wrapper=wrapper,
            args=args,
            runtime_meta=runtime_meta,
        ),
        checkpoint_out,
    )

    audit = correction_parameter_audit(model)
    final_a = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    final_b = wrapper.correction.B_theta.weight.detach().cpu().contiguous()

    report = {
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_TRAINING_REPORT_V1",
        "result": "PASS_GEN5_AINIT_RNG_CAUSAL_TRAINING_CELL",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "recovery_implementation_authority_commit":
            RECOVERY_IMPLEMENTATION_AUTHORITY_COMMIT,
        "failed_execution_authority_commit": FAILED_EXECUTION_AUTHORITY_COMMIT,
        "mechanism_freeze_commit": MECHANISM_FREEZE_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "a_init_seed": args.a_init_seed,
        "training_rng_seed": args.training_rng_seed,
        "cell_name": cell,
        "a_init_sha256": runtime_meta["a_init_sha256"],
        "a_init_torch_version": runtime_meta["a_init_torch_version"],
        "a_init_authentication": runtime_meta["a_init_authentication"],
        "cross_version_fixed_hash_authority":
            runtime_meta["cross_version_fixed_hash_authority"],
        "arm": ARM,
        "pressure": PRESSURE,
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
        "loss_decreased_from_step0": p3a.phase3a_loss_decreased_from_step0(
            losses[0],
            post_step20_loss,
        ),
        "gradient_norms_before_clip": grad_norms,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "dev_final_3way_ce": dev_ce,
        "dev_accuracy": dev_accuracy,
        "contention_geometry": geometry,
        "trainable_tensor_names": audit["trainable_names"],
        "trainable_tensor_count": audit["trainable_tensor_count"],
        "trainable_numel": audit["trainable_numel"],
        "final_A_theta_sha256": tensor_sha256(final_a),
        "final_B_theta_sha256": tensor_sha256(final_b),
        "parent_signature_before": parent_before,
        "parent_signature_after": parent_parameter_fingerprint(model),
        "runtime": runtime_meta["runtime"],
        "train_stream_chunks": TRAIN_ROWS // BACKBONE_STREAM_ROWS,
        "post_step20_stream_chunks": post_chunks,
        "dev_stream_chunks": dev_chunks,
        "final_correction_file_sha256": sha256_file(checkpoint_out),
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
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_TRAINING_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "a_init_seed": args.a_init_seed,
        "training_rng_seed": args.training_rng_seed,
        "cell_name": cell,
        "a_init_sha256": runtime_meta["a_init_sha256"],
        "a_init_torch_version": runtime_meta["a_init_torch_version"],
        "a_init_authentication": runtime_meta["a_init_authentication"],
        "cross_version_fixed_hash_authority":
            runtime_meta["cross_version_fixed_hash_authority"],
        "pressure": PRESSURE,
        "arm": ARM,
        "training_report_sha256": sha256_file(
            cell_dir / "training_report.json"
        ),
        "final_correction_file_sha256": sha256_file(checkpoint_out),
        "optimizer_step_count": TOTAL_OPTIMIZER_STEPS,
        "training_executed": True,
        "backward_executed": True,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(cell_dir / "run_provenance.json", provenance)

    print(
        "GEN5_AINIT_RNG_CAUSAL_CELL_PASS "
        f"cell={cell} "
        f"dev_ce={dev_ce:.12g} "
        f"dev_accuracy={dev_accuracy:.12g}"
    )
    return report


def _validate_two_t4s() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        torch.cuda.device_count() == MATRIX_GPU_COUNT,
        f"GPU_COUNT:{torch.cuda.device_count()}",
    )
    devices = []
    for index in range(MATRIX_GPU_COUNT):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        require(name == "Tesla T4", f"GPU_NAME:{index}:{name}")
        require(
            capability == (7, 5),
            f"GPU_CAPABILITY:{index}:{capability}",
        )
        devices.append({
            "index": index,
            "name": name,
            "capability": list(capability),
        })
    return {
        "gpu_count": MATRIX_GPU_COUNT,
        "devices": devices,
    }


def _cell_command(
    args: argparse.Namespace,
    *,
    a_init_seed: int,
    training_rng_seed: int,
    scratch_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-cell",
        "--expected-head", args.expected_head,
        "--implementation-freeze-commit",
        str(args.implementation_freeze_commit),
        "--execution-authority-commit",
        str(args.execution_authority_commit),
        "--a-init-seed", str(a_init_seed),
        "--training-rng-seed", str(training_rng_seed),
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
    cells: Sequence[tuple[int, int]],
    scratch_root: Path,
    log_path: Path,
) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
    with log_path.open("w", encoding="utf-8") as log:
        for a_init_seed, training_rng_seed in cells:
            completed = subprocess.run(
                _cell_command(
                    args,
                    a_init_seed=a_init_seed,
                    training_rng_seed=training_rng_seed,
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


def _namespace_from_dict(
    values: Mapping[str, Any],
) -> argparse.Namespace:
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
    cells: Sequence[tuple[int, int]],
    scratch_root: str,
    log_path: str,
    args_dict: Mapping[str, Any],
) -> int:
    return _run_worker(
        _namespace_from_dict(args_dict),
        worker_index=worker_index,
        cells=cells,
        scratch_root=Path(scratch_root),
        log_path=Path(log_path),
    )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_contract()
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    gpu_meta = _validate_two_t4s()

    output_root = Path(args.output_root)
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")

    scratch_parent = Path(
        tempfile.mkdtemp(prefix="contramamba_gen5_ainit_rng_")
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
    for worker_index, cells in enumerate(MATRIX_WORKER_CELLS):
        command = [
            sys.executable,
            "-c",
            (
                "from scripts.train_reason_router_gen5_ainit_rng_"
                "causal_intervention import _matrix_worker_entry; "
                "raise SystemExit(_matrix_worker_entry("
                f"{worker_index!r}, {list(cells)!r}, "
                f"{str(worker_roots[worker_index])!r}, "
                f"{str(worker_logs[worker_index])!r}, "
                f"{args_payload!r}))"
            ),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        processes.append(
            subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                text=True,
            )
        )

    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        output_root.mkdir(parents=True, exist_ok=False)
        fail_dir = output_root / "failed_worker_logs"
        fail_dir.mkdir()
        for src in worker_logs:
            if src.exists():
                shutil.copy2(src, fail_dir / src.name)
        raise AInitRNGCausalInterventionError(
            f"MATRIX_WORKER_FAILURE:{return_codes}"
        )

    output_root.mkdir(parents=True, exist_ok=False)
    success_log_dir = output_root / "worker_logs"
    success_log_dir.mkdir()

    reports = []
    for worker_index, worker_root in enumerate(worker_roots):
        if worker_logs[worker_index].exists():
            shutil.copy2(
                worker_logs[worker_index],
                success_log_dir / worker_logs[worker_index].name,
            )
        for a_init_seed, training_rng_seed in MATRIX_WORKER_CELLS[worker_index]:
            cell = cell_name(a_init_seed, training_rng_seed)
            source = worker_root / cell
            require(source.is_dir(), f"WORKER_CELL_MISSING:{cell}")
            destination = output_root / cell
            shutil.copytree(source, destination)
            reports.append(
                json.loads(
                    (destination / "training_report.json")
                    .read_text(encoding="utf-8")
                )
            )

    reports.sort(
        key=lambda row: (
            int(row["a_init_seed"]),
            int(row["training_rng_seed"]),
        )
    )
    require(len(reports) == 6, "MATRIX_CELL_COUNT")

    summary = {
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_MATRIX_SUMMARY_V1",
        "result": "PASS_GEN5_AINIT_RNG_CAUSAL_SIX_OFFDIAGONAL_MATRIX",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "gpu_topology": GPU_TOPOLOGY,
        "gpu_meta": gpu_meta,
        "worker_cells": [
            [cell_name(a, r) for a, r in worker]
            for worker in MATRIX_WORKER_CELLS
        ],
        "new_cell_count": 6,
        "reused_diagonal_cell_count": 3,
        "cells": reports,
        "pressure": PRESSURE,
        "arm": ARM,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer_steps_per_cell": TOTAL_OPTIMIZER_STEPS,
        "optimizer_step_count": 6 * TOTAL_OPTIMIZER_STEPS,
        "training_executed": True,
        "backward_executed": True,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    summary_path = output_root / "matrix_summary.json"
    _write_json_once(summary_path, summary)

    provenance = {
        "schema_version": "GEN5_AINIT_RNG_CAUSAL_MATRIX_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "matrix_summary_sha256": sha256_file(summary_path),
        "gpu_topology": GPU_TOPOLOGY,
        "new_cell_count": 6,
        "reused_diagonal_cell_count": 3,
        "optimizer_step_count": 6 * TOTAL_OPTIMIZER_STEPS,
        "training_executed": True,
        "backward_executed": True,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(
        output_root / "run_provenance.json",
        provenance,
    )

    shutil.rmtree(scratch_parent)

    print("GEN5_AINIT_RNG_CAUSAL_SIX_OFFDIAGONAL_MATRIX_PASS")
    print("NEW_CELLS=6")
    print("REUSED_DIAGONAL_CELLS=3")
    print("GPU_WORKERS=2")
    print(f"GPU_TOPOLOGY={GPU_TOPOLOGY}")
    print(
        "WORKER0="
        + ",".join(cell_name(*cell) for cell in MATRIX_WORKER_CELLS[0])
    )
    print(
        "WORKER1="
        + ",".join(cell_name(*cell) for cell in MATRIX_WORKER_CELLS[1])
    )
    print("PRESSURE=P0")
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_NEW_OPTIMIZER_STEPS=120")
    print("TRAINING_EXECUTED=True")
    print("BACKWARD_EXECUTED=True")
    print("TASK_EVALUATION_EXECUTED=True")
    print("CONFIRMATORY_9601_9900_LOADED=False")
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
    parser.add_argument("--a-init-seed", type=int)
    parser.add_argument("--training-rng-seed", type=int)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    runtime_fields = (
        "implementation_freeze_commit",
        "execution_authority_commit",
        "a_init_seed",
        "training_rng_seed",
        "tokenizer_snapshot",
        "model_snapshot",
        "checkpoint",
        "preflight_output",
        "output_root",
    )

    if args.static_verify_only:
        for field in runtime_fields:
            require(
                getattr(args, field) is None,
                f"STATIC_RUNTIME_ARG_FORBIDDEN:{field}",
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
        require(
            args.a_init_seed is not None
            and args.training_rng_seed is not None,
            "PREFLIGHT_CELL_REQUIRED",
        )
        validate_offdiagonal_cell(
            args.a_init_seed,
            args.training_rng_seed,
        )
        require(
            args.preflight_output is not None,
            "PREFLIGHT_OUTPUT_REQUIRED",
        )
        require(args.output_root is None, "PREFLIGHT_OUTPUT_ROOT_FORBIDDEN")
    elif args.run_cell:
        require(
            args.a_init_seed is not None
            and args.training_rng_seed is not None,
            "RUN_CELL_REQUIRED",
        )
        validate_offdiagonal_cell(
            args.a_init_seed,
            args.training_rng_seed,
        )
        require(args.output_root is not None, "RUN_OUTPUT_ROOT_REQUIRED")
        require(
            args.preflight_output is None,
            "RUN_PREFLIGHT_OUTPUT_FORBIDDEN",
        )
    elif args.run_matrix:
        require(args.a_init_seed is None, "MATRIX_A_INIT_SEED_FORBIDDEN")
        require(
            args.training_rng_seed is None,
            "MATRIX_TRAINING_RNG_SEED_FORBIDDEN",
        )
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
