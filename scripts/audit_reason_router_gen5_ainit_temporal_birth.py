#!/usr/bin/env python3
"""Gen5 A-init representation-freedom temporal-birth audit.

Phase A implemented here replays the exact frozen 20-step Phase3A P0
optimization trajectory for the full 3x3 A-init x training-RNG grid and
captures compact A/B/gradient trajectory evidence.

Phase B raw-write functional-birth analysis is intentionally not executed by
this Phase-A implementation. It is authorized by the same frozen authority but
must be implemented/executed only after successful Phase-A collection/import
and final-checkpoint replay authentication.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contramamba.gen5_phase2_state_update_ownership import (  # noqa: E402
    correction_optimizer_parameters,
    parent_parameter_fingerprint,
    phase2_final_three_way_ce,
)
from scripts import train_reason_router_gen5_ainit_rng_causal_intervention as base  # noqa: E402
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
AUTHORITY_COMMIT = "20ae761dbff10ad70853b10910cbe12e51e0666a"
REPLAY_AUTHENTICATION_CORRECTION_COMMIT = (
    "341e2e59668ea9b575007ddf591424b831170577"
)
SOURCE_EVIDENCE_FREEZE_COMMIT = "d53c33b5a64e02b4f439a1f6b283b07990296bf8"
AUTHORITY_PATH = (
    "reports/reason_router_gen5_ainit_representation_freedom_"
    "temporal_birth_audit_authority_spec_candidate.md"
)
REPLAY_AUTHENTICATION_CORRECTION_PATH = (
    "reports/reason_router_gen5_ainit_representation_freedom_"
    "temporal_birth_replay_authentication_correction_spec_candidate.md"
)

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py",
    "tests/test_reason_router_gen5_ainit_rng_causal_intervention.py",
    "scripts/audit_reason_router_gen5_ainit_temporal_birth.py",
    "tests/test_reason_router_gen5_ainit_temporal_birth.py",
})

FACTOR_SEEDS = (6201, 6202, 6203)
FULL_FACTORIAL_CELLS = tuple(
    (a_init_seed, training_rng_seed)
    for a_init_seed in FACTOR_SEEDS
    for training_rng_seed in FACTOR_SEEDS
)
GPU_QUEUES = {
    0: (
        (6201, 6201),
        (6201, 6203),
        (6202, 6202),
        (6203, 6201),
        (6203, 6203),
    ),
    1: (
        (6201, 6202),
        (6202, 6201),
        (6202, 6203),
        (6203, 6202),
    ),
}
GPU_TOPOLOGY = "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"

REPLAY_AUTH_RECOVERY_CELLS = {
    0: (6201, 6201),
    1: (6201, 6202),
}
HISTORICAL_TRAINING_REPORT_SHA256 = {
    (6201, 6201): "aaedd2a51439e96ee7c2c3a1681a66d35ef31ffca85de22a944de3fb16dda7cf",
    (6201, 6202): "1c39f125b8d297ac9004c827ba48bba5f77dc45f393499a9c5e9d3dd5f590744",
    (6201, 6203): "63a0a30d16ed17da04aea130145053ba529da9d7b9d08fe0c94c5d31fce62d36",
    (6202, 6201): "117418bc673caf82468909f816077dd28c8c7a205a134f97d0a168b7d38e653f",
    (6202, 6202): "7c4159c54d003454849d9f1a74de8dc1b75f649173d0f77ec1e889133c4bd98e",
    (6202, 6203): "a4dc7fc1f271b32e3118d70bd4c0aa4236fda791467ed797a67e01f98cd7ab40",
    (6203, 6201): "5a73154cbfe9ac999757d3032c927e37ebb20220fa45f36ef736cad12faf3f76",
    (6203, 6202): "5c9412f0ece129fbf0dda9827ed59553209807999d704842c61329528cc80e0b",
    (6203, 6203): "74deb7ddeff8fdecbd1895ce43c4506934df720eeb618458337ea9acbd1c07b5",
}

EPS32 = float(torch.finfo(torch.float32).eps)
SCALAR_TRACE_EPS_MULTIPLIER = 32.0
PARAMETER_MAX_ABS_BOUND = 1.0e-5
PARAMETER_RELATIVE_L2_BOUND = 1.0e-4
OPERATOR_RELATIVE_FROBENIUS_BOUND = 1.0e-4

PRESSURE = "P0"
ARM = "G5-C0"
TRAIN_ROWS = 3360
DEV_ROWS = 840
SPLIT_SEED = 16384
TOTAL_OPTIMIZER_STEPS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
PREFLIGHT_CELL = (6201, 6201)
FROZEN_SNAPSHOT_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)

REPLAY_RUN_PREFIX = (
    "reports/reason_router_gen5_ainit_temporal_birth_replay_runs/"
)

OFFDIAGONAL_RUN_ROOT = (
    "reports/reason_router_gen5_ainit_rng_causal_intervention_runs/"
    "gen5-ainit-rng-causal-six-offdiag-87f8255-r1"
)
DIAGONAL_RUN_ROOT = (
    "reports/reason_router_gen5_phase3a_training_runs/"
    "gen5-phase3a-contention-qualification-9cell-d58e894-retry3"
)

FROZEN_FINAL_SOURCES: dict[tuple[int, int], tuple[str, str, str]] = {
    (6201, 6201): (
        f"{DIAGONAL_RUN_ROOT}/cells/seed6201/P0/final_correction.pt",
        "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
        "GEN5_PHASE3A_FINAL_CORRECTION_V1",
    ),
    (6201, 6202): (
        f"{OFFDIAGONAL_RUN_ROOT}/A6201-R6202/final_correction.pt",
        "ef03fbbedf3fab8efb255f2ccb6ec33cdb65ee6881a92e6b4d43fe1e719e40d4",
        "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
    ),
    (6201, 6203): (
        f"{OFFDIAGONAL_RUN_ROOT}/A6201-R6203/final_correction.pt",
        "15582eda034befb1c8d202f04c494f7fd5882bf9fd60ddc963232761057b3df7",
        "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
    ),
    (6202, 6201): (
        f"{OFFDIAGONAL_RUN_ROOT}/A6202-R6201/final_correction.pt",
        "3c217b43eb164583980bee91c39d53a7a3dc20d181e0341db35ffe8ece162ed3",
        "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
    ),
    (6202, 6202): (
        f"{DIAGONAL_RUN_ROOT}/cells/seed6202/P0/final_correction.pt",
        "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
        "GEN5_PHASE3A_FINAL_CORRECTION_V1",
    ),
    (6202, 6203): (
        f"{OFFDIAGONAL_RUN_ROOT}/A6202-R6203/final_correction.pt",
        "d1c478b448c5f53e2a97552196a454f63f9d9081de614a6d9a5cd4e389e30ee2",
        "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
    ),
    (6203, 6201): (
        f"{OFFDIAGONAL_RUN_ROOT}/A6203-R6201/final_correction.pt",
        "d9e1062baf554867b212da57c9d30fe8efeccafc77255ba869affac691606359",
        "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
    ),
    (6203, 6202): (
        f"{OFFDIAGONAL_RUN_ROOT}/A6203-R6202/final_correction.pt",
        "32943df3558ef72eb7f7b7bfb70c6a1a03185a1ea6ca4dd83b9677cd59d13698",
        "GEN5_AINIT_RNG_CAUSAL_FINAL_CORRECTION_V1",
    ),
    (6203, 6203): (
        f"{DIAGONAL_RUN_ROOT}/cells/seed6203/P0/final_correction.pt",
        "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
        "GEN5_PHASE3A_FINAL_CORRECTION_V1",
    ),
}


class TemporalBirthError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TemporalBirthError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TemporalBirthError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def status_paths() -> set[str]:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return base.tensor_sha256(value)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def cell_name(a_init_seed: int, training_rng_seed: int) -> str:
    return f"A{a_init_seed}-R{training_rng_seed}"


def validate_full_factorial_cell(
    a_init_seed: int,
    training_rng_seed: int,
) -> None:
    base.validate_factor_cell(a_init_seed, training_rng_seed)
    require(
        (a_init_seed, training_rng_seed) in FULL_FACTORIAL_CELLS,
        f"CELL_NOT_FULL_FACTORIAL:{a_init_seed}:{training_rng_seed}",
    )


def validate_gpu_queues() -> None:
    flattened = [
        cell
        for worker_id in (0, 1)
        for cell in GPU_QUEUES[worker_id]
    ]
    require(len(flattened) == 9, "GPU_QUEUE_CELL_COUNT")
    require(len(set(flattened)) == 9, "GPU_QUEUE_DUPLICATE")
    require(set(flattened) == set(FULL_FACTORIAL_CELLS), "GPU_QUEUE_MATRIX")
    require(len(GPU_QUEUES[0]) == 5, "GPU0_CELL_COUNT")
    require(len(GPU_QUEUES[1]) == 4, "GPU1_CELL_COUNT")


def validate_frozen_contract() -> None:
    require(FACTOR_SEEDS == base.FACTOR_SEEDS == (6201, 6202, 6203), "FACTOR_SEEDS")
    require(PRESSURE == base.PRESSURE == "P0", "PRESSURE")
    require(ARM == base.ARM == "G5-C0", "ARM")
    require(TRAIN_ROWS == base.TRAIN_ROWS == p3a.TRAIN_ROWS == 3360, "TRAIN_ROWS")
    require(DEV_ROWS == base.DEV_ROWS == p3a.DEV_ROWS == 840, "DEV_ROWS")
    require(SPLIT_SEED == base.SPLIT_SEED == p3a.SPLIT_SEED == 16384, "SPLIT_SEED")
    require(
        TOTAL_OPTIMIZER_STEPS
        == base.TOTAL_OPTIMIZER_STEPS
        == p3a.TOTAL_OPTIMIZER_STEPS
        == 20,
        "OPTIMIZER_STEPS",
    )
    require(LEARNING_RATE == base.LEARNING_RATE == p3a.LEARNING_RATE == 0.001, "LR")
    require(WEIGHT_DECAY == base.WEIGHT_DECAY == p3a.WEIGHT_DECAY == 0.0001, "WD")
    require(
        GRADIENT_CLIP_NORM
        == base.GRADIENT_CLIP_NORM
        == p3a.GRADIENT_CLIP_NORM
        == 5.0,
        "CLIP",
    )
    require(not base.CONFIRMATORY_9601_9900_ALLOWED, "CONFIRMATORY_ALLOWED")
    validate_gpu_queues()


def authenticate_repo(
    expected_head: str,
    *,
    allow_implementation_worktree: bool,
) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH:{branch}")
    require(head == expected_head, f"HEAD:{head}")
    require(
        git_rc("merge-base", "--is-ancestor", AUTHORITY_COMMIT, expected_head) == 0,
        "AUTHORITY_NOT_ANCESTOR",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            SOURCE_EVIDENCE_FREEZE_COMMIT,
            expected_head,
        )
        == 0,
        "SOURCE_EVIDENCE_NOT_ANCESTOR",
    )

    require(
        git_rc(
            "merge-base", "--is-ancestor",
            REPLAY_AUTHENTICATION_CORRECTION_COMMIT, expected_head,
        ) == 0,
        "REPLAY_AUTHENTICATION_CORRECTION_NOT_ANCESTOR",
    )

    authority_blob = git("rev-parse", f"{AUTHORITY_COMMIT}:{AUTHORITY_PATH}")
    live_authority_blob = git("rev-parse", f"HEAD:{AUTHORITY_PATH}")
    require(authority_blob == live_authority_blob, "AUTHORITY_BLOB_DRIFT")
    correction_blob = git(
        "rev-parse",
        f"{REPLAY_AUTHENTICATION_CORRECTION_COMMIT}:"
        f"{REPLAY_AUTHENTICATION_CORRECTION_PATH}",
    )
    live_correction_blob = git(
        "rev-parse", f"HEAD:{REPLAY_AUTHENTICATION_CORRECTION_PATH}"
    )
    require(
        correction_blob == live_correction_blob,
        "REPLAY_AUTHENTICATION_CORRECTION_BLOB_DRIFT",
    )

    observed = status_paths()
    if allow_implementation_worktree:
        require(
            observed <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(observed)}",
        )
    else:
        require(not observed, f"WORKTREE_NOT_CLEAN:{sorted(observed)}")


def validate_runtime_authority(
    *,
    expected_head: str,
    implementation_freeze_commit: str,
) -> None:
    require(
        implementation_freeze_commit == expected_head,
        "IMPLEMENTATION_FREEZE_MUST_EQUAL_EXECUTION_HEAD",
    )
    text = git("show", f"{AUTHORITY_COMMIT}:{AUTHORITY_PATH}")
    correction_text = git(
        "show",
        f"{REPLAY_AUTHENTICATION_CORRECTION_COMMIT}:"
        f"{REPLAY_AUTHENTICATION_CORRECTION_PATH}",
    )
    required_tokens = (
        "COMBINED_IMPLEMENTATION_AND_EXECUTION_AUTHORITY=YES_CONDITIONAL",
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_ONLY_AFTER_IMPLEMENTATION_VALIDATION_AND_FREEZE",
        "TRAINING_ALLOWED=YES_EXACT_HISTORICAL_20_STEP_REPLAY_ONLY",
        "OBJECTIVE_CHANGE_ALLOWED=NO",
        "LEARNING_RATE_CHANGE_ALLOWED=NO",
        "WEIGHT_DECAY_CHANGE_ALLOWED=NO",
        "GRADIENT_CLIP_CHANGE_ALLOWED=NO",
        "STEP_COUNT_CHANGE_ALLOWED=NO",
        "DATA_OR_SPLIT_CHANGE_ALLOWED=NO",
        "LABEL_CHANGE_ALLOWED=NO",
        "PARENT_PARAMETER_UPDATE_ALLOWED=NO",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "PREFLIGHT_COLLECTION=FORBIDDEN",
        "FAILED_RUN_COLLECTION=FORBIDDEN",
        f"GPU_TOPOLOGY={GPU_TOPOLOGY}",
        "SCIENTIFIC_GPU_COUNT=2",
        "CROSS_GPU_SCIENTIFIC_TENSOR_REDUCTION=FORBIDDEN",
    )
    for token in required_tokens:
        require(token in text, f"AUTHORITY_TOKEN:{token}")

    correction_tokens = (
        "AUTHORITY_CORRECTION=YES",
        "32 * EPS32 * max(1, abs(historical))",
        "max_abs <= 1.0e-5",
        "relative_l2 <= 1.0e-4",
        "operator_relative_frobenius_residual <= 1.0e-4",
        "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_NUMERICAL_REPLAY_AUTHENTICATION_PASS",
        "failed numerical replay: `DO_NOT_COLLECT`",
        "successful corrected Phase A replay: `COLLECT_AND_IMPORT_REQUIRED`",
    )
    for token in correction_tokens:
        require(
            token in correction_text,
            f"REPLAY_AUTHENTICATION_CORRECTION_TOKEN:{token}",
        )


def load_frozen_final_checkpoint(
    a_init_seed: int,
    training_rng_seed: int,
) -> dict[str, Any]:
    validate_full_factorial_cell(a_init_seed, training_rng_seed)
    relative_path, expected_file_sha, expected_schema = FROZEN_FINAL_SOURCES[
        (a_init_seed, training_rng_seed)
    ]
    path = ROOT / relative_path
    require(path.is_file(), f"FROZEN_CHECKPOINT_MISSING:{relative_path}")
    observed_file_sha = sha256_file(path)
    require(
        observed_file_sha == expected_file_sha,
        (
            f"FROZEN_CHECKPOINT_SHA:{cell_name(a_init_seed, training_rng_seed)}:"
            f"{observed_file_sha}"
        ),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, Mapping), "FROZEN_CHECKPOINT_PAYLOAD")
    require(payload.get("schema_version") == expected_schema, "FROZEN_CHECKPOINT_SCHEMA")
    require(payload.get("arm") == ARM, "FROZEN_CHECKPOINT_ARM")
    require(payload.get("pressure") == PRESSURE, "FROZEN_CHECKPOINT_PRESSURE")

    if a_init_seed == training_rng_seed:
        require(int(payload.get("seed", -1)) == a_init_seed, "DIAGONAL_SEED")
    else:
        require(int(payload.get("a_init_seed", -1)) == a_init_seed, "OFFDIAGONAL_A_SEED")
        require(
            int(payload.get("training_rng_seed", -1)) == training_rng_seed,
            "OFFDIAGONAL_R_SEED",
        )

    state = payload.get("state_dict")
    require(isinstance(state, Mapping), "FROZEN_CHECKPOINT_STATE")
    a = state.get("A_theta.weight")
    b = state.get("B_theta.weight")
    require(torch.is_tensor(a) and tuple(a.shape) == (2, 768), "FROZEN_A_SHAPE")
    require(torch.is_tensor(b) and tuple(b.shape) == (24576, 2), "FROZEN_B_SHAPE")
    a = a.detach().cpu().contiguous()
    b = b.detach().cpu().contiguous()
    hashes = payload.get("tensor_sha256") or {}
    require(hashes.get("A_theta.weight") == tensor_sha256(a), "FROZEN_A_TENSOR_SHA")
    require(hashes.get("B_theta.weight") == tensor_sha256(b), "FROZEN_B_TENSOR_SHA")

    return {
        "relative_path": relative_path,
        "file_sha256": observed_file_sha,
        "schema_version": expected_schema,
        "A_theta.weight": a,
        "B_theta.weight": b,
        "A_theta_sha256": tensor_sha256(a),
        "B_theta_sha256": tensor_sha256(b),
    }


def load_historical_training_trace(
    a_init_seed: int,
    training_rng_seed: int,
) -> dict[str, Any]:
    cell = (a_init_seed, training_rng_seed)
    validate_full_factorial_cell(*cell)
    frozen_relative = Path(FROZEN_FINAL_SOURCES[cell][0])
    report_relative = frozen_relative.with_name("training_report.json")
    report_path = ROOT / report_relative
    require(
        report_path.is_file(),
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:"
            f"HISTORICAL_REPORT_MISSING:{report_relative}"
        ),
    )
    observed_sha = sha256_file(report_path)
    expected_sha = HISTORICAL_TRAINING_REPORT_SHA256[cell]
    require(
        observed_sha == expected_sha,
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:HISTORICAL_REPORT_SHA:"
            f"{cell_name(*cell)}:expected={expected_sha}:observed={observed_sha}"
        ),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    losses = report.get("training_losses")
    grad_norms = report.get("gradient_norms_before_clip")
    require(
        isinstance(losses, list) and len(losses) == TOTAL_OPTIMIZER_STEPS,
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:HISTORICAL_LOSS_TRACE:"
            f"{cell_name(*cell)}"
        ),
    )
    require(
        isinstance(grad_norms, list)
        and len(grad_norms) == TOTAL_OPTIMIZER_STEPS,
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:HISTORICAL_GRAD_TRACE:"
            f"{cell_name(*cell)}"
        ),
    )
    return {
        "relative_path": str(report_relative).replace("\\", "/"),
        "file_sha256": observed_sha,
        "training_losses": [float(v) for v in losses],
        "gradient_norms_before_clip": [float(v) for v in grad_norms],
    }


def scalar_trace_tolerance(reference: float) -> float:
    return (
        SCALAR_TRACE_EPS_MULTIPLIER
        * EPS32
        * max(1.0, abs(float(reference)))
    )


def scalar_trace_diagnostics(
    *,
    observed: float,
    historical: float,
) -> dict[str, Any]:
    absolute_error = abs(float(observed) - float(historical))
    tolerance = scalar_trace_tolerance(historical)
    return {
        "observed": float(observed),
        "historical": float(historical),
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "pass": absolute_error <= tolerance,
    }


def tensor_replay_diagnostics(
    replay: torch.Tensor,
    historical: torch.Tensor,
) -> dict[str, Any]:
    replay_cpu = replay.detach().cpu().contiguous()
    historical_cpu = historical.detach().cpu().contiguous()
    require(
        replay_cpu.shape == historical_cpu.shape,
        "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:PARAMETER_SHAPE",
    )
    delta = replay_cpu.to(torch.float64) - historical_cpu.to(torch.float64)
    max_abs = float(delta.abs().max().item())
    l2 = float(torch.linalg.vector_norm(delta).item())
    historical_l2 = float(
        torch.linalg.vector_norm(historical_cpu.to(torch.float64)).item()
    )
    relative_l2 = l2 / max(historical_l2, EPS32)
    max_abs_pass = max_abs <= PARAMETER_MAX_ABS_BOUND
    relative_l2_pass = relative_l2 <= PARAMETER_RELATIVE_L2_BOUND
    return {
        "torch_equal": torch.equal(replay_cpu, historical_cpu),
        "replay_sha256": tensor_sha256(replay_cpu),
        "historical_sha256": tensor_sha256(historical_cpu),
        "max_abs": max_abs,
        "l2": l2,
        "historical_l2": historical_l2,
        "relative_l2": relative_l2,
        "max_abs_bound": PARAMETER_MAX_ABS_BOUND,
        "relative_l2_bound": PARAMETER_RELATIVE_L2_BOUND,
        "max_abs_pass": max_abs_pass,
        "relative_l2_pass": relative_l2_pass,
        "pass": max_abs_pass and relative_l2_pass,
    }


def operator_replay_diagnostics(
    *,
    replay_a: torch.Tensor,
    replay_b: torch.Tensor,
    historical_a: torch.Tensor,
    historical_b: torch.Tensor,
) -> dict[str, Any]:
    distance = math.sqrt(max(
        0.0,
        operator_distance_sq(
            replay_a, replay_b, historical_a, historical_b
        ),
    ))
    historical_norm = math.sqrt(max(
        0.0,
        operator_norm_sq(historical_a, historical_b),
    ))
    relative = distance / max(historical_norm, EPS32)
    return {
        "frobenius_distance": distance,
        "historical_frobenius_norm": historical_norm,
        "relative_frobenius_residual": relative,
        "relative_frobenius_bound": OPERATOR_RELATIVE_FROBENIUS_BOUND,
        "pass": relative <= OPERATOR_RELATIVE_FROBENIUS_BOUND,
    }


def operator_inner(
    a_left: torch.Tensor,
    b_left: torch.Tensor,
    a_right: torch.Tensor,
    b_right: torch.Tensor,
) -> float:
    a1 = a_left.detach().cpu().to(torch.float64)
    b1 = b_left.detach().cpu().to(torch.float64)
    a2 = a_right.detach().cpu().to(torch.float64)
    b2 = b_right.detach().cpu().to(torch.float64)
    require(a1.ndim == 2 and a2.ndim == 2, "OPERATOR_A_NDIM")
    require(b1.ndim == 2 and b2.ndim == 2, "OPERATOR_B_NDIM")
    require(a1.shape[0] == b1.shape[1], "OPERATOR_LEFT_RANK")
    require(a2.shape[0] == b2.shape[1], "OPERATOR_RIGHT_RANK")
    require(b1.shape[0] == b2.shape[0], "OPERATOR_OUTPUT_DIM")
    require(a1.shape[1] == a2.shape[1], "OPERATOR_INPUT_DIM")
    cross_b = b1.T @ b2
    cross_a = a2 @ a1.T
    return float(torch.trace(cross_b @ cross_a).item())


def operator_norm_sq(a: torch.Tensor, b: torch.Tensor) -> float:
    value = operator_inner(a, b, a, b)
    return max(0.0, value)


def operator_distance_sq(
    a_left: torch.Tensor,
    b_left: torch.Tensor,
    a_right: torch.Tensor,
    b_right: torch.Tensor,
) -> float:
    value = (
        operator_norm_sq(a_left, b_left)
        + operator_norm_sq(a_right, b_right)
        - 2.0 * operator_inner(a_left, b_left, a_right, b_right)
    )
    return max(0.0, value)


def operator_snapshot_stats(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, Any]:
    a64 = a.detach().cpu().to(torch.float64)
    b64 = b.detach().cpu().to(torch.float64)
    _qb, rb = torch.linalg.qr(b64, mode="reduced")
    _qa, ra = torch.linalg.qr(a64.T, mode="reduced")
    singular = torch.linalg.svdvals(rb @ ra.T)
    singular_values = [float(v) for v in singular.tolist()]
    fro_sq = operator_norm_sq(a64, b64)
    fro = math.sqrt(fro_sq)
    spectral = max(singular_values, default=0.0)
    return {
        "frobenius_norm": fro,
        "frobenius_norm_sq": fro_sq,
        "spectral_norm": spectral,
        "nonzero_singular_values": singular_values,
    }


def snapshot_state(
    *,
    wrapper: Any,
    a_init: torch.Tensor,
    t: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    a = wrapper.correction.A_theta.weight.detach().cpu().contiguous().clone()
    b = wrapper.correction.B_theta.weight.detach().cpu().contiguous().clone()
    a0 = a_init.detach().cpu().to(dtype=a.dtype).contiguous()
    a_displacement = float(torch.linalg.vector_norm(a - a0).item())
    b_norm = float(torch.linalg.vector_norm(b).item())
    op = operator_snapshot_stats(a, b)
    public = {
        "t": t,
        "A_theta_sha256": tensor_sha256(a),
        "B_theta_sha256": tensor_sha256(b),
        "A_theta_norm": float(torch.linalg.vector_norm(a).item()),
        "A_init_displacement": a_displacement,
        "B_theta_norm": b_norm,
        "BA": op,
    }
    private = {
        "t": t,
        "A_theta.weight": a,
        "B_theta.weight": b,
        "metrics": public,
    }
    return public, private


def _operator_gram(
    states: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    cells = list(FULL_FACTORIAL_CELLS)
    gram = torch.empty((9, 9), dtype=torch.float64)
    for i, left in enumerate(cells):
        for j, right in enumerate(cells):
            if j < i:
                gram[i, j] = gram[j, i]
                continue
            value = operator_inner(
                states[left]["A"],
                states[left]["B"],
                states[right]["A"],
                states[right]["B"],
            )
            gram[i, j] = value
            gram[j, i] = value
    return cells, gram


def _tensor_gram(
    values: Mapping[tuple[int, int], torch.Tensor],
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    cells = list(FULL_FACTORIAL_CELLS)
    vectors = {
        cell: values[cell].detach().cpu().to(torch.float64).reshape(-1)
        for cell in cells
    }
    gram = torch.empty((9, 9), dtype=torch.float64)
    for i, left in enumerate(cells):
        for j, right in enumerate(cells):
            if j < i:
                gram[i, j] = gram[j, i]
                continue
            value = float(torch.dot(vectors[left], vectors[right]).item())
            gram[i, j] = value
            gram[j, i] = value
    return cells, gram


def _quadratic_norm(coeff: torch.Tensor, gram: torch.Tensor) -> float:
    value = float((coeff @ gram @ coeff).item())
    return max(0.0, value)


def factorial_energy_from_gram(
    cells: Sequence[tuple[int, int]],
    gram: torch.Tensor,
) -> dict[str, Any]:
    require(len(cells) == 9, "FACTORIAL_CELL_COUNT")
    index = {cell: i for i, cell in enumerate(cells)}
    grand = torch.full((9,), 1.0 / 9.0, dtype=torch.float64)

    ss_a = 0.0
    for a in FACTOR_SEEDS:
        mean_a = torch.zeros(9, dtype=torch.float64)
        for r in FACTOR_SEEDS:
            mean_a[index[(a, r)]] = 1.0 / 3.0
        ss_a += 3.0 * _quadratic_norm(mean_a - grand, gram)

    ss_r = 0.0
    for r in FACTOR_SEEDS:
        mean_r = torch.zeros(9, dtype=torch.float64)
        for a in FACTOR_SEEDS:
            mean_r[index[(a, r)]] = 1.0 / 3.0
        ss_r += 3.0 * _quadratic_norm(mean_r - grand, gram)

    ss_ar = 0.0
    ss_total = 0.0
    for a, r in cells:
        unit = torch.zeros(9, dtype=torch.float64)
        unit[index[(a, r)]] = 1.0

        mean_a = torch.zeros(9, dtype=torch.float64)
        mean_r = torch.zeros(9, dtype=torch.float64)
        for rr in FACTOR_SEEDS:
            mean_a[index[(a, rr)]] = 1.0 / 3.0
        for aa in FACTOR_SEEDS:
            mean_r[index[(aa, r)]] = 1.0 / 3.0

        ss_total += _quadratic_norm(unit - grand, gram)
        ss_ar += _quadratic_norm(unit - mean_a - mean_r + grand, gram)

    closure_abs = abs(ss_total - (ss_a + ss_r + ss_ar))
    fractions = None
    if ss_total > 0.0:
        fractions = {
            "A": ss_a / ss_total,
            "R": ss_r / ss_total,
            "AR": ss_ar / ss_total,
        }
    return {
        "SS_A": ss_a,
        "SS_R": ss_r,
        "SS_AR": ss_ar,
        "SS_TOTAL": ss_total,
        "closure_abs": closure_abs,
        "normalized_fractions": fractions,
        "A_over_R_energy_ratio": None if ss_r == 0.0 else ss_a / ss_r,
    }


def grouped_pair_stats_from_gram(
    cells: Sequence[tuple[int, int]],
    gram: torch.Tensor,
) -> dict[str, Any]:
    index = {cell: i for i, cell in enumerate(cells)}
    classes = {
        "same_training_rng_different_a": [],
        "same_a_different_training_rng": [],
    }
    for i, left in enumerate(cells):
        for right in cells[i + 1 :]:
            same_a = left[0] == right[0]
            same_r = left[1] == right[1]
            if same_r and not same_a:
                classes["same_training_rng_different_a"].append((left, right))
            elif same_a and not same_r:
                classes["same_a_different_training_rng"].append((left, right))

    result: dict[str, Any] = {}
    for label, pairs in classes.items():
        rows = []
        for left, right in pairs:
            li = index[left]
            ri = index[right]
            left_norm = max(0.0, float(gram[li, li].item()))
            right_norm = max(0.0, float(gram[ri, ri].item()))
            distance_sq = max(
                0.0,
                left_norm + right_norm - 2.0 * float(gram[li, ri].item()),
            )
            denom_sq = 0.5 * (left_norm + right_norm)
            normalized = None
            if denom_sq > 0.0:
                normalized = math.sqrt(distance_sq / denom_sq)
            rows.append({
                "left": cell_name(*left),
                "right": cell_name(*right),
                "squared_distance": distance_sq,
                "normalized_residual": normalized,
            })

        require(len(rows) == 9, f"PAIR_COUNT:{label}:{len(rows)}")
        normalized_values = [
            row["normalized_residual"]
            for row in rows
            if row["normalized_residual"] is not None
        ]
        result[label] = {
            "count": len(rows),
            "mean_squared_distance": sum(
                row["squared_distance"] for row in rows
            )
            / len(rows),
            "mean_normalized_residual": (
                None
                if not normalized_values
                else sum(normalized_values) / len(normalized_values)
            ),
            "normalized_defined_count": len(normalized_values),
            "pairs": rows,
        }

    numerator = result["same_training_rng_different_a"]["mean_squared_distance"]
    denominator = result["same_a_different_training_rng"]["mean_squared_distance"]
    result["A_over_R_pair_distance_ratio"] = (
        None if denominator == 0.0 else numerator / denominator
    )
    return result


def trajectory_factor_metrics(
    cells: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for t in range(TOTAL_OPTIMIZER_STEPS + 1):
        states: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for a, r in FULL_FACTORIAL_CELLS:
            cell = cells[cell_name(a, r)]
            snap = cell["snapshots"][t]
            require(int(snap["t"]) == t, f"SNAPSHOT_T:{cell_name(a, r)}:{t}")
            states[(a, r)] = {
                "A": snap["A_theta.weight"],
                "B": snap["B_theta.weight"],
            }
        ordered, gram = _operator_gram(states)
        rows.append({
            "t": t,
            "grouped": grouped_pair_stats_from_gram(ordered, gram),
            "factorial": factorial_energy_from_gram(ordered, gram),
        })
    return rows


def step0_grad_b_factor_metrics(
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values = {
        (a, r): cells[cell_name(a, r)]["step0_grad_B"]
        for a, r in FULL_FACTORIAL_CELLS
    }
    ordered, gram = _tensor_gram(values)
    return {
        "grouped": grouped_pair_stats_from_gram(ordered, gram),
        "factorial": factorial_energy_from_gram(ordered, gram),
    }


def _validate_two_t4s() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 2, f"CUDA_DEVICE_COUNT:{torch.cuda.device_count()}")
    devices = []
    for index in (0, 1):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        require(name == "Tesla T4", f"GPU_NAME:{index}:{name}")
        require(capability == (7, 5), f"GPU_CAPABILITY:{index}:{capability}")
        devices.append({
            "index": index,
            "name": name,
            "capability": list(capability),
        })
    return {"gpu_count": 2, "devices": devices}


def run_static_verify(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=args.allow_opening_worktree,
    )
    validate_frozen_contract()
    validate_phase_b_static_contract()

    loaded = {}
    for a, r in FULL_FACTORIAL_CELLS:
        source = load_frozen_final_checkpoint(a, r)
        loaded[cell_name(a, r)] = {
            "relative_path": source["relative_path"],
            "file_sha256": source["file_sha256"],
            "schema_version": source["schema_version"],
            "A_theta_sha256": source["A_theta_sha256"],
            "B_theta_sha256": source["B_theta_sha256"],
        }

    print("GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"AUTHORITY_COMMIT={AUTHORITY_COMMIT}")
    print(
        "REPLAY_AUTHENTICATION_CORRECTION_COMMIT="
        f"{REPLAY_AUTHENTICATION_CORRECTION_COMMIT}"
    )
    print(f"SCALAR_TRACE_EPS_MULTIPLIER={SCALAR_TRACE_EPS_MULTIPLIER:.17g}")
    print(f"PARAMETER_MAX_ABS_BOUND={PARAMETER_MAX_ABS_BOUND:.17g}")
    print(f"PARAMETER_RELATIVE_L2_BOUND={PARAMETER_RELATIVE_L2_BOUND:.17g}")
    print(
        "OPERATOR_RELATIVE_FROBENIUS_BOUND="
        f"{OPERATOR_RELATIVE_FROBENIUS_BOUND:.17g}"
    )
    print("CELLS=9")
    print("GPU_WORKERS=2")
    print(f"GPU_TOPOLOGY={GPU_TOPOLOGY}")
    print("GPU0=" + ",".join(cell_name(*cell) for cell in GPU_QUEUES[0]))
    print("GPU1=" + ",".join(cell_name(*cell) for cell in GPU_QUEUES[1]))
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_REPLAY_OPTIMIZER_STEPS=180")
    print("PHASE_A_IMPLEMENTED=True")
    print("PHASE_B_IMPLEMENTED=True")
    print("PHASE_B_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    for name in sorted(loaded):
        print(f"FROZEN_CHECKPOINT={name}:{loaded[name]['file_sha256']}")


def _prepare_runtime(args: argparse.Namespace) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    static, encoded, snapshot, checkpoint_path = base._prepare_runtime_inputs(args)
    require(len(static["train_rows"]) == TRAIN_ROWS, "RUNTIME_TRAIN_ROWS")
    require(len(static["dev_rows"]) == DEV_ROWS, "RUNTIME_DEV_ROWS")
    return static, encoded, snapshot, checkpoint_path


def run_cuda_preflight(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_runtime_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    validate_frozen_contract()
    _validate_two_t4s()

    a_seed, r_seed = PREFLIGHT_CELL
    static, encoded, snapshot, checkpoint_path = _prepare_runtime(args)
    del static
    model, wrapper, runtime_meta, strong_mask, planes = base._prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        a_init_seed=a_seed,
        training_rng_seed=r_seed,
    )
    parent_before = runtime_meta["parent_before"]

    features, labels, active, targets = p3a._feature_batch_to_device(
        encoded["train_bundle"],
        torch.device("cuda:0"),
    )
    optimizer_parameters = correction_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    a_init = base.reconstruct_a_init(a_seed)
    _public0, private0 = snapshot_state(
        wrapper=wrapper,
        a_init=a_init,
        t=0,
    )
    b0 = private0["B_theta.weight"]
    require(int(torch.count_nonzero(b0).item()) == 0, "PREFLIGHT_B0_NONZERO")

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(r_seed)
    torch.cuda.manual_seed_all(r_seed)

    optimizer.zero_grad(set_to_none=True)
    output, chunks = p3a._streamed_forward(
        model,
        features,
        active,
        targets,
        pressure=PRESSURE,
        strong_mask=strong_mask,
        planes=planes,
    )
    require(chunks == TRAIN_ROWS // base.BACKBONE_STREAM_ROWS, "PREFLIGHT_CHUNKS")
    logits = output["logits"]
    loss = phase2_final_three_way_ce(logits, labels)
    require(bool(torch.isfinite(loss).item()), "PREFLIGHT_LOSS_NONFINITE")
    loss.backward()

    a_grad = wrapper.correction.A_theta.weight.grad
    b_grad = wrapper.correction.B_theta.weight.grad
    require(a_grad is not None and b_grad is not None, "PREFLIGHT_GRAD_MISSING")
    require(bool(torch.isfinite(a_grad).all()), "PREFLIGHT_A_GRAD_NONFINITE")
    require(bool(torch.isfinite(b_grad).all()), "PREFLIGHT_B_GRAD_NONFINITE")
    a_grad_nonzero = int(torch.count_nonzero(a_grad).item())
    require(a_grad_nonzero == 0, f"PREFLIGHT_A_GRAD_NONZERO:{a_grad_nonzero}")
    b_grad_norm = float(torch.linalg.vector_norm(b_grad).detach().cpu().item())
    require(b_grad_norm > 0.0, "PREFLIGHT_B_GRAD_ZERO")

    parent_grads = [
        name
        for name, parameter in model.named_parameters()
        if ".correction." not in name and parameter.grad is not None
    ]
    require(not parent_grads, f"PREFLIGHT_PARENT_GRADIENT:{parent_grads[:5]}")

    total_preclip = torch.nn.utils.clip_grad_norm_(
        optimizer_parameters,
        GRADIENT_CLIP_NORM,
    )
    require(bool(torch.isfinite(total_preclip).item()), "PREFLIGHT_CLIP_NONFINITE")
    optimizer.step()

    public1, _private1 = snapshot_state(
        wrapper=wrapper,
        a_init=a_init,
        t=1,
    )
    require(parent_parameter_fingerprint(model) == parent_before, "PREFLIGHT_PARENT_MUTATION")

    print("GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_CUDA_PREFLIGHT_PASS")
    print(f"CELL={cell_name(a_seed, r_seed)}")
    print("B0_EXACT_ZERO=True")
    print("GRAD_A0_EXACT_ZERO=True")
    print(f"GRAD_B0_NORM={b_grad_norm:.17g}")
    print(f"TOTAL_GRAD_NORM_PRECLIP={float(total_preclip.detach().cpu().item()):.17g}")
    print(f"A1_INIT_DISPLACEMENT={public1['A_init_displacement']:.17g}")
    print(f"B1_NORM={public1['B_theta_norm']:.17g}")
    print("OPTIMIZER_STEP_COUNT=1")
    print("PREFLIGHT_COLLECTION=FORBIDDEN")
    print("CONFIRMATORY_9601_9900_LOADED=False")


def _run_replay_cell(
    *,
    args: argparse.Namespace,
    a_init_seed: int,
    training_rng_seed: int,
    encoded: Mapping[str, Any],
    snapshot: Path,
    checkpoint_path: Path,
    historical_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_full_factorial_cell(a_init_seed, training_rng_seed)
    frozen = load_frozen_final_checkpoint(a_init_seed, training_rng_seed)

    model, wrapper, runtime_meta, strong_mask, planes = base._prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        a_init_seed=a_init_seed,
        training_rng_seed=training_rng_seed,
    )
    parent_before = runtime_meta["parent_before"]
    features, labels, active, targets = p3a._feature_batch_to_device(
        encoded["train_bundle"],
        torch.device("cuda:0"),
    )

    optimizer_parameters = correction_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(training_rng_seed)
    torch.cuda.manual_seed_all(training_rng_seed)

    a_init = base.reconstruct_a_init(a_init_seed)
    snapshot_public: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    public0, private0 = snapshot_state(wrapper=wrapper, a_init=a_init, t=0)
    snapshot_public.append(public0)
    snapshots.append(private0)
    require(
        torch.equal(private0["A_theta.weight"], a_init),
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:A0_RECONSTRUCTION:"
            f"{cell_name(a_init_seed, training_rng_seed)}"
        ),
    )
    require(
        int(torch.count_nonzero(private0["B_theta.weight"]).item()) == 0,
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:B0_NONZERO:"
            f"{cell_name(a_init_seed, training_rng_seed)}"
        ),
    )

    losses: list[float] = []
    gradient_metrics: list[dict[str, Any]] = []
    step0_grad_a: torch.Tensor | None = None
    step0_grad_b: torch.Tensor | None = None

    for step in range(TOTAL_OPTIMIZER_STEPS):
        # Preserve the historical CUDA sequence through the explicit sync.
        optimizer.zero_grad(set_to_none=True)
        output, chunks = p3a._streamed_forward(
            model,
            features,
            active,
            targets,
            pressure=PRESSURE,
            strong_mask=strong_mask,
            planes=planes,
        )
        require(
            chunks == TRAIN_ROWS // base.BACKBONE_STREAM_ROWS,
            f"CHUNKS:{cell_name(a_init_seed, training_rng_seed)}:{step}",
        )
        logits = output["logits"]
        require(tuple(logits.shape) == (TRAIN_ROWS, 3), "TRAIN_LOGIT_SHAPE")
        require(bool(torch.isfinite(logits).all()), "TRAIN_LOGIT_NONFINITE")
        loss = phase2_final_three_way_ce(logits, labels)
        require(bool(torch.isfinite(loss).item()), f"LOSS_NONFINITE:{step}")
        loss.backward()

        a_grad = wrapper.correction.A_theta.weight.grad
        b_grad = wrapper.correction.B_theta.weight.grad
        require(a_grad is not None and b_grad is not None, f"GRAD_MISSING:{step}")
        require(bool(torch.isfinite(a_grad).all()), f"A_GRAD_NONFINITE:{step}")
        require(bool(torch.isfinite(b_grad).all()), f"B_GRAD_NONFINITE:{step}")

        parent_grads = [
            name
            for name, parameter in model.named_parameters()
            if ".correction." not in name and parameter.grad is not None
        ]
        require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")

        clipped = torch.nn.utils.clip_grad_norm_(optimizer_parameters, GRADIENT_CLIP_NORM)
        require(bool(torch.isfinite(clipped).item()), "GRAD_NORM_NONFINITE")
        optimizer.step()
        require(
            bool(torch.isfinite(wrapper.correction.A_theta.weight).all())
            and bool(torch.isfinite(wrapper.correction.B_theta.weight).all()),
            "CORRECTION_PARAMETER_NONFINITE",
        )

        # Match historical scalar transfers and sync before instrumentation.
        loss_value = float(loss.detach().cpu().item())
        total_preclip = float(clipped.detach().cpu().item())
        del output, logits, loss
        torch.cuda.synchronize()

        require(
            total_preclip < GRADIENT_CLIP_NORM,
            f"ACTIVE_CLIP_PREVENTS_POSTSYNC_GRAD_CAPTURE:{cell_name(a_init_seed, training_rng_seed)}:{step}:{total_preclip:.17g}",
        )
        a_grad_cpu = a_grad.detach().cpu().contiguous().clone()
        b_grad_cpu = b_grad.detach().cpu().contiguous().clone()
        a_grad_norm = float(torch.linalg.vector_norm(a_grad_cpu).item())
        b_grad_norm = float(torch.linalg.vector_norm(b_grad_cpu).item())

        if step == 0:
            step0_grad_a = a_grad_cpu.clone()
            step0_grad_b = b_grad_cpu.clone()
            require(
                int(torch.count_nonzero(step0_grad_a).item()) == 0,
                (
                    "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:"
                    "STEP0_A_GRAD_NONZERO:"
                    f"{cell_name(a_init_seed, training_rng_seed)}"
                ),
            )
            require(
                bool(torch.isfinite(step0_grad_b).all())
                and float(torch.linalg.vector_norm(step0_grad_b).item()) > 0.0,
                (
                    "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:"
                    "STEP0_B_GRAD_INVALID:"
                    f"{cell_name(a_init_seed, training_rng_seed)}"
                ),
            )

        losses.append(loss_value)
        gradient_metrics.append({
            "step_preupdate": step,
            "grad_A_norm_preclip": a_grad_norm,
            "grad_B_norm_preclip": b_grad_norm,
            "total_grad_norm_preclip": total_preclip,
        })

        public, private = snapshot_state(wrapper=wrapper, a_init=a_init, t=step + 1)
        snapshot_public.append(public)
        snapshots.append(private)

        if historical_trace is not None:
            expected_loss = float(historical_trace["training_losses"][step])
            expected_grad = float(
                historical_trace["gradient_norms_before_clip"][step]
            )
            loss_auth = scalar_trace_diagnostics(
                observed=loss_value, historical=expected_loss
            )
            grad_auth = scalar_trace_diagnostics(
                observed=total_preclip, historical=expected_grad
            )
            gradient_metrics[-1][
                "historical_loss_authentication"
            ] = loss_auth
            gradient_metrics[-1][
                "historical_total_grad_authentication"
            ] = grad_auth
            if not loss_auth["pass"] or not grad_auth["pass"]:
                raise TemporalBirthError(
                    (
                        "TEMPORAL_REPLAY_SCALAR_AUTHENTICATION_FAILED:"
                        f"{cell_name(a_init_seed, training_rng_seed)}:"
                        f"step={step}:"
                        f"loss_error={loss_auth['absolute_error']:.17g}:"
                        f"loss_tolerance={loss_auth['tolerance']:.17g}:"
                        f"grad_error={grad_auth['absolute_error']:.17g}:"
                        f"grad_tolerance={grad_auth['tolerance']:.17g}"
                    )
                )

    require(len(snapshots) == 21, "SNAPSHOT_COUNT")
    require(len(losses) == 20, "LOSS_COUNT")
    require(len(gradient_metrics) == 20, "GRADIENT_METRIC_COUNT")
    require(step0_grad_a is not None and step0_grad_b is not None, "STEP0_GRAD_CAPTURE")

    final_a = snapshots[-1]["A_theta.weight"]
    final_b = snapshots[-1]["B_theta.weight"]
    frozen_a = frozen["A_theta.weight"]
    frozen_b = frozen["B_theta.weight"]

    a_auth = tensor_replay_diagnostics(final_a, frozen_a)
    b_auth = tensor_replay_diagnostics(final_b, frozen_b)
    if not a_auth["pass"] or not b_auth["pass"]:
        raise TemporalBirthError(
            (
                "TEMPORAL_REPLAY_PARAMETER_AUTHENTICATION_FAILED:"
                f"{cell_name(a_init_seed, training_rng_seed)}:"
                f"A_max_abs={a_auth['max_abs']:.17g}:"
                f"A_relative_l2={a_auth['relative_l2']:.17g}:"
                f"B_max_abs={b_auth['max_abs']:.17g}:"
                f"B_relative_l2={b_auth['relative_l2']:.17g}"
            )
        )

    operator_auth = operator_replay_diagnostics(
        replay_a=final_a,
        replay_b=final_b,
        historical_a=frozen_a,
        historical_b=frozen_b,
    )
    if not operator_auth["pass"]:
        raise TemporalBirthError(
            (
                "TEMPORAL_REPLAY_OPERATOR_AUTHENTICATION_FAILED:"
                f"{cell_name(a_init_seed, training_rng_seed)}:"
                "relative_frobenius_residual="
                f"{operator_auth['relative_frobenius_residual']:.17g}:"
                "bound="
                f"{operator_auth['relative_frobenius_bound']:.17g}"
            )
        )

    require(
        parent_parameter_fingerprint(model) == parent_before,
        (
            "TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED:PARENT_MUTATION:"
            f"{cell_name(a_init_seed, training_rng_seed)}"
        ),
    )

    result = {
        "a_init_seed": a_init_seed,
        "training_rng_seed": training_rng_seed,
        "cell_name": cell_name(a_init_seed, training_rng_seed),
        "snapshots_public": snapshot_public,
        "snapshots": snapshots,
        "training_losses": losses,
        "gradient_metrics": gradient_metrics,
        "step0_grad_A": step0_grad_a,
        "step0_grad_B": step0_grad_b,
        "step0_grad_A_sha256": tensor_sha256(step0_grad_a),
        "step0_grad_B_sha256": tensor_sha256(step0_grad_b),
        "frozen_final_checkpoint": {
            "relative_path": frozen["relative_path"],
            "file_sha256": frozen["file_sha256"],
            "A_theta_sha256": frozen["A_theta_sha256"],
            "B_theta_sha256": frozen["B_theta_sha256"],
        },
        "historical_training_trace": {
            "relative_path": (
                historical_trace["relative_path"]
                if historical_trace is not None else None
            ),
            "file_sha256": (
                historical_trace["file_sha256"]
                if historical_trace is not None else None
            ),
            "all_scalar_steps_authenticated": historical_trace is not None,
        },
        "final_replay_authentication": {
            "A": a_auth,
            "B": b_auth,
            "operator": operator_auth,
            "numerically_authenticated": True,
        },
        "parent_signature_before": parent_before,
        "parent_signature_after": parent_parameter_fingerprint(model),
        "runtime": runtime_meta["runtime"],
    }

    del model, wrapper
    torch.cuda.empty_cache()
    return result



def run_replay_auth_recovery_worker(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_runtime_authority(expected_head=args.expected_head, implementation_freeze_commit=args.implementation_freeze_commit)
    validate_frozen_contract()
    require(args.worker_id in (0, 1), f"RECOVERY_WORKER_ID:{args.worker_id}")
    a_seed, r_seed = REPLAY_AUTH_RECOVERY_CELLS[args.worker_id]
    trace = load_historical_training_trace(a_seed, r_seed)
    _static, encoded, snapshot, checkpoint_path = _prepare_runtime(args)
    _run_replay_cell(
        args=args,
        a_init_seed=a_seed,
        training_rng_seed=r_seed,
        encoded=encoded,
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        historical_trace=trace,
    )
    print(
        "GEN5_AINIT_TEMPORAL_BIRTH_REPLAY_AUTH_RECOVERY_WORKER_PASS "
        f"worker={args.worker_id} cell={cell_name(a_seed, r_seed)}"
    )


def run_replay_auth_recovery(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_runtime_authority(expected_head=args.expected_head, implementation_freeze_commit=args.implementation_freeze_commit)
    validate_frozen_contract()
    _validate_two_t4s()
    scratch_root = Path(tempfile.mkdtemp(prefix="contramamba_gen5_replay_auth_recovery_"))
    logs = [scratch_root / f"worker{worker_id}.log" for worker_id in (0, 1)]
    common = [
        sys.executable, str(Path(__file__).resolve()),
        "--replay-auth-recovery-worker",
        "--expected-head", args.expected_head,
        "--implementation-freeze-commit", args.implementation_freeze_commit,
        "--checkpoint", str(args.checkpoint),
    ]
    if args.model_snapshot is not None:
        common += ["--model-snapshot", str(args.model_snapshot)]
    if args.tokenizer_snapshot is not None:
        common += ["--tokenizer-snapshot", str(args.tokenizer_snapshot)]
    processes, handles = [], []
    try:
        for worker_id in (0, 1):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
            handle = logs[worker_id].open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(
                common + ["--worker-id", str(worker_id)],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True,
            ))
        return_codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    try:
        for worker_id, log_path in enumerate(logs):
            print(f"=== REPLAY AUTH RECOVERY WORKER {worker_id} LOG ===")
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8"), end="")
        if any(code != 0 for code in return_codes):
            raise TemporalBirthError(f"REPLAY_AUTH_RECOVERY_FAILED:{return_codes}")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    print("GEN5_AINIT_TEMPORAL_BIRTH_REPLAY_AUTH_RECOVERY_PASS")
    print("CELLS=A6201-R6201,A6201-R6202")
    print("GPU_WORKERS=2")
    print(f"GPU_TOPOLOGY={GPU_TOPOLOGY}")
    print("HISTORICAL_SCALAR_TRACE_NUMERICALLY_AUTHENTICATED=True")
    print("FINAL_A_B_NUMERICALLY_AUTHENTICATED=True")
    print("FINAL_OPERATOR_NUMERICALLY_AUTHENTICATED=True")
    print("RECOVERY_COLLECTION=FORBIDDEN")
    print("SCIENTIFIC_INTERPRETATION=FORBIDDEN")


def run_replay_worker(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_runtime_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    validate_frozen_contract()
    require(args.worker_id in (0, 1), f"WORKER_ID:{args.worker_id}")
    require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")

    _static, encoded, snapshot, checkpoint_path = _prepare_runtime(args)
    worker_root = Path(args.scratch_root) / f"worker{args.worker_id}"
    require(not worker_root.exists(), f"WORKER_OUTPUT_COLLISION:{worker_root}")
    worker_root.mkdir(parents=True, exist_ok=False)

    cells: dict[str, Any] = {}
    public_cells: list[dict[str, Any]] = []
    for a_seed, r_seed in GPU_QUEUES[args.worker_id]:
        historical_trace = load_historical_training_trace(a_seed, r_seed)
        cell = _run_replay_cell(
            args=args,
            a_init_seed=a_seed,
            training_rng_seed=r_seed,
            encoded=encoded,
            snapshot=snapshot,
            checkpoint_path=checkpoint_path,
            historical_trace=historical_trace,
        )
        name = cell["cell_name"]
        cells[name] = {
            "a_init_seed": cell["a_init_seed"],
            "training_rng_seed": cell["training_rng_seed"],
            "snapshots": cell["snapshots"],
            "step0_grad_A": cell["step0_grad_A"],
            "step0_grad_B": cell["step0_grad_B"],
            "training_losses": cell["training_losses"],
            "gradient_metrics": cell["gradient_metrics"],
        }
        public_cells.append({
            "a_init_seed": cell["a_init_seed"],
            "training_rng_seed": cell["training_rng_seed"],
            "cell_name": name,
            "snapshots": cell["snapshots_public"],
            "training_losses": cell["training_losses"],
            "gradient_metrics": cell["gradient_metrics"],
            "step0_grad_A_sha256": cell["step0_grad_A_sha256"],
            "step0_grad_B_sha256": cell["step0_grad_B_sha256"],
            "frozen_final_checkpoint": cell["frozen_final_checkpoint"],
            "historical_training_trace": cell["historical_training_trace"],
            "final_replay_authentication": cell["final_replay_authentication"],
            "parent_signature_before": cell["parent_signature_before"],
            "parent_signature_after": cell["parent_signature_after"],
            "runtime": cell["runtime"],
        })

    torch.save(
        {
            "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_WORKER_TRAJECTORY_V1",
            "worker_id": args.worker_id,
            "queue": list(GPU_QUEUES[args.worker_id]),
            "cells": cells,
        },
        worker_root / "worker_trajectory.pt",
    )
    (worker_root / "worker_summary.json").write_bytes(
        canonical_json_bytes({
            "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_WORKER_SUMMARY_V1",
            "worker_id": args.worker_id,
            "queue": [
                cell_name(a, r)
                for a, r in GPU_QUEUES[args.worker_id]
            ],
            "cells": public_cells,
            "train_encoding_sha256": encoded["train_encoding_sha256"],
            "dev_encoding_sha256": encoded["dev_encoding_sha256"],
            "resolved_model_snapshot": str(snapshot),
            "frozen_snapshot_revision": FROZEN_SNAPSHOT_REVISION,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "training_executed": True,
            "optimizer_step_count": len(public_cells) * TOTAL_OPTIMIZER_STEPS,
            "task_evaluation_executed": False,
            "confirmatory_9601_9900_loaded": False,
        })
    )
    print(
        "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_WORKER_PASS "
        f"worker={args.worker_id} cells={len(public_cells)}"
    )


def run_replay_matrix(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_runtime_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    validate_frozen_contract()
    gpu_meta = _validate_two_t4s()
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    output_root = Path(args.output_root)
    require(
        str(output_root).replace("\\", "/").startswith(REPLAY_RUN_PREFIX),
        f"OUTPUT_ROOT_PREFIX:{output_root}",
    )
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")

    scratch_root = Path(tempfile.mkdtemp(prefix="contramamba_gen5_temporal_birth_"))
    logs = [
        scratch_root / f"worker{worker_id}.log"
        for worker_id in (0, 1)
    ]

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-replay-worker",
        "--expected-head", args.expected_head,
        "--implementation-freeze-commit", args.implementation_freeze_commit,
        "--checkpoint", str(args.checkpoint),
        "--scratch-root", str(scratch_root),
    ]
    if args.model_snapshot is not None:
        common += ["--model-snapshot", str(args.model_snapshot)]
    if args.tokenizer_snapshot is not None:
        common += ["--tokenizer-snapshot", str(args.tokenizer_snapshot)]

    processes = []
    handles = []
    try:
        for worker_id in (0, 1):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
            handle = logs[worker_id].open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(
                common + ["--worker-id", str(worker_id)],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)

        return_codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()

    if any(code != 0 for code in return_codes):
        for worker_id, log_path in enumerate(logs):
            print(f"=== TEMPORAL BIRTH WORKER {worker_id} FAILURE LOG ===")
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8"), end="")
        shutil.rmtree(scratch_root, ignore_errors=True)
        raise TemporalBirthError(f"REPLAY_WORKER_FAILURE:{return_codes}")

    merged_cells: dict[str, Any] = {}
    public_cells: list[dict[str, Any]] = []
    worker_summaries = []
    for worker_id in (0, 1):
        worker_root = scratch_root / f"worker{worker_id}"
        worker_trajectory = torch.load(
            worker_root / "worker_trajectory.pt",
            map_location="cpu",
            weights_only=True,
        )
        summary = json.loads(
            (worker_root / "worker_summary.json").read_text(encoding="utf-8")
        )
        worker_summaries.append(summary)
        for name, cell in worker_trajectory["cells"].items():
            require(name not in merged_cells, f"DUPLICATE_CELL:{name}")
            merged_cells[name] = cell
        public_cells.extend(summary["cells"])

    require(len(worker_summaries) == 2, "WORKER_SUMMARY_COUNT")
    require(
        len({row["train_encoding_sha256"] for row in worker_summaries}) == 1,
        "TRAIN_ENCODING_WORKER_MISMATCH",
    )
    require(
        len({row["dev_encoding_sha256"] for row in worker_summaries}) == 1,
        "DEV_ENCODING_WORKER_MISMATCH",
    )
    require(
        len({row["resolved_model_snapshot"] for row in worker_summaries}) == 1,
        "MODEL_SNAPSHOT_WORKER_MISMATCH",
    )
    train_encoding_sha256 = worker_summaries[0]["train_encoding_sha256"]
    dev_encoding_sha256 = worker_summaries[0]["dev_encoding_sha256"]
    resolved_model_snapshot = worker_summaries[0]["resolved_model_snapshot"]

    require(len(merged_cells) == 9, f"MERGED_CELL_COUNT:{len(merged_cells)}")
    require(
        set(merged_cells)
        == {cell_name(a, r) for a, r in FULL_FACTORIAL_CELLS},
        "MERGED_CELL_SET",
    )
    public_cells.sort(
        key=lambda row: (
            int(row["a_init_seed"]),
            int(row["training_rng_seed"]),
        )
    )

    factor_trajectory = trajectory_factor_metrics(merged_cells)
    grad_b_factor = step0_grad_b_factor_metrics(merged_cells)

    output_root.mkdir(parents=True, exist_ok=False)
    log_root = output_root / "worker_logs"
    log_root.mkdir()
    for worker_id, log_path in enumerate(logs):
        shutil.copy2(log_path, log_root / f"worker{worker_id}.log")

    trajectory_path = output_root / "temporal_birth_trajectory.pt"
    torch.save(
        {
            "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_TRAJECTORY_V2",
            "authority_commit": AUTHORITY_COMMIT,
            "replay_authentication_correction_commit": (
                REPLAY_AUTHENTICATION_CORRECTION_COMMIT
            ),
            "execution_head": args.expected_head,
            "implementation_freeze_commit": args.implementation_freeze_commit,
            "factor_seeds": FACTOR_SEEDS,
            "time_axis": list(range(TOTAL_OPTIMIZER_STEPS + 1)),
            "gpu_queues": GPU_QUEUES,
            "cells": merged_cells,
        },
        trajectory_path,
    )

    summary = {
        "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_REPLAY_SUMMARY_V2",
        "result": (
            "PASS_GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_"
            "NUMERICAL_REPLAY_AUTHENTICATION"
        ),
        "authority_commit": AUTHORITY_COMMIT,
        "replay_authentication_correction_commit": (
            REPLAY_AUTHENTICATION_CORRECTION_COMMIT
        ),
        "source_evidence_freeze_commit": SOURCE_EVIDENCE_FREEZE_COMMIT,
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "factor_seeds": list(FACTOR_SEEDS),
        "cell_count": 9,
        "time_axis": list(range(TOTAL_OPTIMIZER_STEPS + 1)),
        "pressure": PRESSURE,
        "arm": ARM,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "train_encoding_sha256": train_encoding_sha256,
        "dev_encoding_sha256": dev_encoding_sha256,
        "resolved_model_snapshot": resolved_model_snapshot,
        "frozen_snapshot_revision": FROZEN_SNAPSHOT_REVISION,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer_steps_per_cell": TOTAL_OPTIMIZER_STEPS,
        "optimizer_step_count": 9 * TOTAL_OPTIMIZER_STEPS,
        "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "gpu_topology": GPU_TOPOLOGY,
        "gpu_queues": {
            str(worker_id): [
                cell_name(a, r)
                for a, r in GPU_QUEUES[worker_id]
            ]
            for worker_id in (0, 1)
        },
        "gpu_runtime": gpu_meta,
        "cells": public_cells,
        "operator_factor_trajectory": factor_trajectory,
        "step0_grad_B_factor_metrics": grad_b_factor,
        "historical_scalar_trace_gate": {
            "eps32": EPS32,
            "eps_multiplier": SCALAR_TRACE_EPS_MULTIPLIER,
        },
        "parameter_replay_gate": {
            "max_abs_bound": PARAMETER_MAX_ABS_BOUND,
            "relative_l2_bound": PARAMETER_RELATIVE_L2_BOUND,
        },
        "operator_replay_gate": {
            "relative_frobenius_bound": OPERATOR_RELATIVE_FROBENIUS_BOUND,
        },
        "all_replays_numerically_authenticated": True,
        "phase_b_executed": False,
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    summary_path = output_root / "temporal_birth_replay_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary))

    provenance = {
        "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_PROVENANCE_V2",
        "status": "PASS",
        "authority_commit": AUTHORITY_COMMIT,
        "authority_blob": git("rev-parse", f"HEAD:{AUTHORITY_PATH}"),
        "replay_authentication_correction_commit": (
            REPLAY_AUTHENTICATION_CORRECTION_COMMIT
        ),
        "replay_authentication_correction_blob": git(
            "rev-parse", f"HEAD:{REPLAY_AUTHENTICATION_CORRECTION_PATH}"
        ),
        "execution_head": args.expected_head,
        "frozen_snapshot_revision": FROZEN_SNAPSHOT_REVISION,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "train_encoding_sha256": train_encoding_sha256,
        "dev_encoding_sha256": dev_encoding_sha256,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "summary_sha256": sha256_file(summary_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "gpu_topology": GPU_TOPOLOGY,
        "cell_count": 9,
        "optimizer_step_count": 9 * TOTAL_OPTIMIZER_STEPS,
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "failed_run_collection_allowed": False,
        "preflight_collection_allowed": False,
    }
    provenance_path = output_root / "run_provenance.json"
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    shutil.rmtree(scratch_root)

    print(
        "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_"
        "NUMERICAL_REPLAY_AUTHENTICATION_PASS"
    )
    print("CELLS=9")
    print("GPU_WORKERS=2")
    print(f"GPU_TOPOLOGY={GPU_TOPOLOGY}")
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_REPLAY_OPTIMIZER_STEPS=180")
    print("ALL_HISTORICAL_SCALAR_TRACES_AUTHENTICATED=True")
    print("ALL_FINAL_PARAMETERS_NUMERICALLY_AUTHENTICATED=True")
    print("ALL_FINAL_OPERATORS_NUMERICALLY_AUTHENTICATED=True")
    print("TASK_EVALUATION_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={summary_path}")
    print(f"TRAJECTORY={trajectory_path}")
    print(f"PROVENANCE={provenance_path}")


# ---------------------------------------------------------------------------
# Phase B: raw-write geometric / functional birth analysis
# ---------------------------------------------------------------------------

PHASE_A_EVIDENCE_FREEZE_COMMIT = "0288a7a77f0b05b7f1fc22dcc878b3cfbf90b29b"
PHASE_A_RUN_NAME = "gen5-ainit-temporal-birth-phase-a-numerical-auth-d940e19-r1"
PHASE_A_RUN_ROOT = (
    "reports/reason_router_gen5_ainit_temporal_birth_replay_runs/"
    + PHASE_A_RUN_NAME
)
PHASE_A_SUMMARY_PATH = PHASE_A_RUN_ROOT + "/temporal_birth_replay_summary.json"
PHASE_A_TRAJECTORY_PATH = PHASE_A_RUN_ROOT + "/temporal_birth_trajectory.pt"
PHASE_A_SUMMARY_SHA256 = (
    "37cab4866c729ad03262df562762ac8b6b91e4e24e2498223483848871ad1df6"
)
PHASE_A_TRAJECTORY_SHA256 = (
    "0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e"
)
PHASE_A_EVIDENCE_REPORT_PATH = (
    "reports/reason_router_gen5_ainit_representation_freedom_"
    "temporal_birth_phase_a_evidence_report_candidate.md"
)

INTERNAL_PRECURSOR_CORRECTION_COMMIT = (
    "e2c563188e9534e898c6ea944c5a05f4056b2fe3"
)
INTERNAL_PRECURSOR_CORRECTION_PATH = (
    "reports/reason_router_gen5_ainit_internal_task_visible_"
    "precursor_localization_authority_correction_spec_candidate.md"
)
INTERNAL_PRECURSOR_CORRECTION_BLOB = (
    "f4a45513712c9a6380b309f0a8171fbeb499a925"
)
INTERNAL_PRECURSOR_METRICS_PATH = (
    "reports/reason_router_gen5_ainit_internal_precursor_localization_runs/"
    "gen5-ainit-internal-precursor-e2c5631-r1/"
    "internal_stage_task_visible_metrics.pt"
)
INTERNAL_PRECURSOR_METRICS_SHA256 = (
    "d00aedc95873ab47716d96773fdb81ee9a3b81c80479b06de38959829e9afa9d"
)

PHASE_B_RUN_PREFIX = (
    "reports/reason_router_gen5_ainit_temporal_birth_analysis_runs/"
)

PHASE_B_CONTROL_DOMAIN = "GEN5_INTERNAL_PRECURSOR_V1"
PHASE_B_CONTROL_COUNT = 8
PHASE_B_PROJECTOR_PINV_RTOL = 1.0e-12
PHASE_B_FUNCTIONAL_BATCH_ROWS = 4
PHASE_B_VALID_TOKEN_COUNT = 60094

PHASE_B_RAW_CONTROL_IDENTITY = (
    (0, 8634153414855414169, "f7d2acc8fcaed199515df9f60642ea4988e7a22a09ad0535af68e63bd358345f"),
    (1, 8942982651256360266, "fc1bdb56e09dcd4adfed107559bf413e0e8aaffdbcbde5f29dc62a3409d89636"),
    (2, 4733603261589309886, "41b1231632a949be7503e9428982bb1ddd24ec221e97a7a8cb39f193a6f1c6c3"),
    (3, 3074483775818615235, "2aaac2b46d5ec9c3733c958079c5fc45d260621a9dd605556c012fd08feb4370"),
    (4, 794199281934087957, "0b05900a008d37156649852ec9d1f9d4e1e86c9d331e30121a99dcc99ffb38ae"),
    (5, 7416946985525545978, "e6ee49fc96278bfa42ac1ebac49d1ea4979c70c155ddb0df5c9c6628790f849b"),
    (6, 259315910244880612, "039946724ac0d0e4b75e12b03f50dae3238ac960102b215ff06e66c0d14f61f5"),
    (7, 883752037080260340, "0c43b7cb9fb25ef4971d053137c50d6ea15c304c9e3d4997bbb4b96fedb05584"),
)

PHASE_B_ENDPOINT_TARGET = {
    "normalized_residual": 0.45922708704045434,
    "task_row_energy": 0.00032369586357429783,
    "control_task_row_energy": 0.00002631725566293209,
    "enrichment": 12.299757532477983,
    "centered_logits": {
        "R_visible": 0.8948354137604684,
        "R_complement": 0.00438080799597352,
        "R_interaction": 0.008403463389073969,
    },
    "two_margins": {
        "R_visible": 0.8949580000020421,
        "R_complement": 0.004406601900046891,
        "R_interaction": 0.008439518433847864,
    },
}
PHASE_B_ENDPOINT_TOLERANCE = {
    "normalized_residual_atol": 5.0e-4,
    "task_row_energy_atol": 5.0e-5,
    "control_task_row_energy_atol": 5.0e-6,
    "enrichment_atol": 5.0e-1,
    "effect_ratio_atol": 5.0e-3,
}
PHASE_B_FULL_REPLAY_ATOL = 5.0e-5
PHASE_B_JOINT_FORWARD_ATOL = 1.0e-6


def phase_b_control_identity(control_index: int) -> dict[str, Any]:
    require(
        0 <= control_index < PHASE_B_CONTROL_COUNT,
        f"PHASE_B_CONTROL_INDEX:{control_index}",
    )
    label = f"{PHASE_B_CONTROL_DOMAIN}|raw_write|{control_index}"
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) & ((1 << 63) - 1)
    expected_index, expected_seed, expected_digest = (
        PHASE_B_RAW_CONTROL_IDENTITY[control_index]
    )
    require(expected_index == control_index, "PHASE_B_CONTROL_INDEX_FREEZE")
    require(seed == expected_seed, f"PHASE_B_CONTROL_SEED:{control_index}")
    require(digest == expected_digest, f"PHASE_B_CONTROL_DIGEST:{control_index}")
    return {
        "index": control_index,
        "seed": seed,
        "sha256_label": digest,
    }


def phase_b_signed_permutation(
    *,
    feature_width: int,
    control_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    require(feature_width > 0, "PHASE_B_CONTROL_FEATURE_WIDTH")
    identity = phase_b_control_identity(control_index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(identity["seed"]))
    permutation = torch.randperm(
        feature_width,
        generator=generator,
        device="cpu",
    )
    sign_bits = torch.randint(
        0,
        2,
        (feature_width,),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    signs = sign_bits.mul(2).sub(1).to(dtype=dtype)
    return (
        permutation.to(device=device),
        signs.to(device=device),
    )


def phase_b_apply_signed_permutation(
    residual: torch.Tensor,
    *,
    control_index: int,
) -> torch.Tensor:
    require(residual.ndim == 3, "PHASE_B_CONTROL_RESIDUAL_RANK")
    permutation, signs = phase_b_signed_permutation(
        feature_width=int(residual.shape[-1]),
        control_index=control_index,
        device=residual.device,
        dtype=residual.dtype,
    )
    return residual.index_select(-1, permutation) * signs.view(1, 1, -1)


def _phase_b_load_phase_a_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = ROOT / PHASE_A_SUMMARY_PATH
    trajectory_path = ROOT / PHASE_A_TRAJECTORY_PATH
    report_path = ROOT / PHASE_A_EVIDENCE_REPORT_PATH
    require(summary_path.is_file(), "PHASE_A_SUMMARY_MISSING")
    require(trajectory_path.is_file(), "PHASE_A_TRAJECTORY_MISSING")
    require(report_path.is_file(), "PHASE_A_EVIDENCE_REPORT_MISSING")
    require(
        sha256_file(summary_path) == PHASE_A_SUMMARY_SHA256,
        "PHASE_A_SUMMARY_SHA",
    )
    require(
        sha256_file(trajectory_path) == PHASE_A_TRAJECTORY_SHA256,
        "PHASE_A_TRAJECTORY_SHA",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(
        summary.get("result")
        == "PASS_GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_NUMERICAL_REPLAY_AUTHENTICATION",
        "PHASE_A_RESULT",
    )
    require(summary.get("cell_count") == 9, "PHASE_A_CELL_COUNT")
    require(
        summary.get("optimizer_step_count") == 180,
        "PHASE_A_OPTIMIZER_STEPS",
    )
    require(
        summary.get("all_replays_numerically_authenticated") is True,
        "PHASE_A_REPLAY_AUTH",
    )
    require(summary.get("phase_b_executed") is False, "PHASE_A_PHASE_B_FLAG")
    require(
        summary.get("confirmatory_9601_9900_loaded") is False,
        "PHASE_A_CONFIRMATORY_FLAG",
    )

    trajectory = torch.load(
        trajectory_path,
        map_location="cpu",
        weights_only=True,
    )
    require(
        trajectory.get("schema_version")
        == "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_TRAJECTORY_V2",
        "PHASE_A_TRAJECTORY_SCHEMA",
    )
    require(
        tuple(trajectory.get("factor_seeds", ())) == FACTOR_SEEDS,
        "PHASE_A_FACTOR_SEEDS",
    )
    require(
        list(trajectory.get("time_axis", ()))
        == list(range(TOTAL_OPTIMIZER_STEPS + 1)),
        "PHASE_A_TIME_AXIS",
    )
    require(
        set(trajectory.get("cells", {}))
        == {cell_name(a, r) for a, r in FULL_FACTORIAL_CELLS},
        "PHASE_A_TRAJECTORY_CELL_SET",
    )
    return summary, trajectory


def _phase_b_git_show_text(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "show", ref],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="strict",
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise TemporalBirthError(f"PHASE_B_GIT_SHOW_FAILURE:{ref}") from exc


def validate_phase_b_static_contract() -> None:
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            PHASE_A_EVIDENCE_FREEZE_COMMIT,
            git("rev-parse", "HEAD"),
        )
        == 0,
        "PHASE_A_EVIDENCE_FREEZE_NOT_ANCESTOR",
    )
    authority_text = _phase_b_git_show_text(
        f"{AUTHORITY_COMMIT}:{AUTHORITY_PATH}"
    )
    required_authority_tokens = (
        "Analyze only:",
        "`raw_write = B_theta A_theta x`",
        "t=0` raw-write residual must be exactly zero",
        "Use the recovered true-forward `joint` analysis-gradient semantics",
        "`SHA256(\"GEN5_INTERNAL_PRECURSOR_V1|raw_write|<control_index>\")`",
        "`0.60 <= R_visible <= 1.40`",
        "`R_complement <= 0.05`",
        "`R_interaction <= 0.05`",
        "`E_task_actual / E_task_control_mean >= 5`",
        "Starting at the already computed geometric-birth step",
        "stop at the first passing step",
        "Successful Phase B scientific run:",
        "`COLLECT_AND_IMPORT_REQUIRED`",
    )
    for token in required_authority_tokens:
        require(token in authority_text, f"PHASE_B_AUTHORITY_TOKEN:{token}")

    correction_blob = git(
        "rev-parse",
        f"{INTERNAL_PRECURSOR_CORRECTION_COMMIT}:"
        f"{INTERNAL_PRECURSOR_CORRECTION_PATH}",
    )
    require(
        correction_blob == INTERNAL_PRECURSOR_CORRECTION_BLOB,
        f"PHASE_B_PRECURSOR_CORRECTION_BLOB:{correction_blob}",
    )

    metrics_path = ROOT / INTERNAL_PRECURSOR_METRICS_PATH
    require(metrics_path.is_file(), "INTERNAL_PRECURSOR_METRICS_MISSING")
    require(
        sha256_file(metrics_path) == INTERNAL_PRECURSOR_METRICS_SHA256,
        "INTERNAL_PRECURSOR_METRICS_SHA",
    )
    for control_index in range(PHASE_B_CONTROL_COUNT):
        phase_b_control_identity(control_index)

    _phase_b_load_phase_a_artifacts()


def _phase_b_snapshot(
    trajectory: Mapping[str, Any],
    cell: tuple[int, int],
    t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    require(0 <= t <= TOTAL_OPTIMIZER_STEPS, f"PHASE_B_T:{t}")
    cell_payload = trajectory["cells"][cell_name(*cell)]
    snapshot = cell_payload["snapshots"][t]
    require(int(snapshot["t"]) == t, f"PHASE_B_SNAPSHOT_T:{cell_name(*cell)}:{t}")
    a = snapshot["A_theta.weight"].detach().cpu().contiguous()
    b = snapshot["B_theta.weight"].detach().cpu().contiguous()
    require(tuple(a.shape) == (2, 768), "PHASE_B_A_SHAPE")
    require(tuple(b.shape) == (24576, 2), "PHASE_B_B_SHAPE")
    return a, b


def _phase_b_data_gram(
    trajectory: Mapping[str, Any],
    covariance: torch.Tensor,
    t: int,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    require(
        covariance.dtype == torch.float64
        and tuple(covariance.shape) == (768, 768),
        "PHASE_B_COVARIANCE",
    )
    cells = list(FULL_FACTORIAL_CELLS)
    gram = torch.empty((9, 9), dtype=torch.float64)
    states = {
        cell: tuple(x.to(torch.float64) for x in _phase_b_snapshot(trajectory, cell, t))
        for cell in cells
    }
    for i, left in enumerate(cells):
        a_left, b_left = states[left]
        for j in range(i, len(cells)):
            right = cells[j]
            a_right, b_right = states[right]
            b_cross = b_left.T @ b_right
            a_cross = a_left @ covariance @ a_right.T
            value = torch.sum(b_cross * a_cross)
            gram[i, j] = value
            gram[j, i] = value
    return cells, gram


def phase_b_geometry_trajectory(
    trajectory: Mapping[str, Any],
    covariance: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for t in range(TOTAL_OPTIMIZER_STEPS + 1):
        cells, gram = _phase_b_data_gram(trajectory, covariance, t)
        rows.append(
            {
                "t": t,
                "grouped": grouped_pair_stats_from_gram(cells, gram),
                "factorial": factorial_energy_from_gram(cells, gram),
            }
        )
    t0 = rows[0]
    require(
        t0["grouped"]["same_training_rng_different_a"]["mean_squared_distance"]
        == 0.0,
        "PHASE_B_T0_PRIMARY_NOT_ZERO",
    )
    require(
        t0["grouped"]["same_a_different_training_rng"]["mean_squared_distance"]
        == 0.0,
        "PHASE_B_T0_CONTROL_NOT_ZERO",
    )
    require(
        t0["factorial"]["SS_TOTAL"] == 0.0,
        "PHASE_B_T0_FACTORIAL_NOT_ZERO",
    )
    return rows


def phase_b_geometric_birth_step(
    geometry: Sequence[Mapping[str, Any]],
) -> int:
    for row in geometry:
        t = int(row["t"])
        if t == 0:
            continue
        distance = float(
            row["grouped"]["same_training_rng_different_a"][
                "mean_squared_distance"
            ]
        )
        if distance > 0.0:
            return t
    raise TemporalBirthError("NO_RAW_WRITE_GEOMETRIC_BIRTH_LOCALIZED")


@contextlib.contextmanager
def _phase_b_joint_analysis_runtime(
    model: torch.nn.Module,
) -> Iterator[None]:
    names = (
        "gradient_ownership_mode",
        "gradient_ownership_lambda",
        "edge_gradient_lambdas",
        "return_q_diagnostics",
    )
    prior = {
        name: (hasattr(model, name), getattr(model, name, None))
        for name in names
    }
    model.gradient_ownership_mode = "joint"
    model.gradient_ownership_lambda = None
    model.edge_gradient_lambdas = None
    model.return_q_diagnostics = True
    try:
        yield
    finally:
        for name in names:
            existed, value = prior[name]
            if existed:
                setattr(model, name, value)
            elif hasattr(model, name):
                delattr(model, name)


def _phase_b_joint_forward_from_hidden(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    hidden_states: torch.Tensor,
) -> Mapping[str, Any]:
    from scripts import (
        reason_router_gen4_six_cell_tier2_inference_adapter as adapter,
    )

    with _phase_b_joint_analysis_runtime(model):
        return model(
            input_ids=None,
            attention_mask=features["attention_mask"],
            claim_mask=features["claim_mask"],
            evidence_mask=features["evidence_mask"],
            decision_mode=adapter.DECISION_MODE,
            gradient_ownership_mode="joint",
            gradient_ownership_lambda=None,
            edge_gradient_lambdas=None,
            return_q_diagnostics=True,
            encoder_hidden_states=hidden_states,
        )


def _phase_b_prepare_common_context(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    stressor_active: torch.Tensor,
    target_indices: torch.Tensor,
    strong_mask: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    require(input_ids.ndim == 2, "PHASE_B_INPUT_RANK")
    require(
        tuple(attention_mask.shape) == tuple(input_ids.shape),
        "PHASE_B_ATTENTION_SHAPE",
    )

    with torch.no_grad():
        hidden = model.mamba.embeddings(input_ids)
        mixer17 = model.mamba.layers[17].mixer
        with p3a.batch_stressor_hook(
            mixer17.in_proj,
            pressure=PRESSURE,
            strong_mask=strong_mask,
            active_rows=stressor_active,
            target_indices=target_indices,
            planes=planes,
        ):
            for layer_index in range(22):
                hidden = model.mamba.layers[layer_index](
                    hidden,
                    cache_params=None,
                    cache_position=None,
                    attention_mask=None,
                )

        residual22 = hidden
        block22 = model.mamba.layers[22]
        mixer_input = block22.norm(
            hidden.to(dtype=block22.norm.weight.dtype)
        )
        native_mixer = wrapper.native_mixer
        native22 = native_mixer(
            mixer_input,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        )

        projected = native_mixer.in_proj(mixer_input).transpose(1, 2)
        hidden_states, gate = projected.chunk(2, dim=1)
        active = attention_mask.to(hidden_states.dtype)
        hidden_states = hidden_states * active.unsqueeze(1)
        conv_hidden = native_mixer.act(
            native_mixer.conv1d(hidden_states)[..., : input_ids.shape[1]]
        )
        conv_hidden = conv_hidden * active.unsqueeze(1)
        ssm_parameters = native_mixer.x_proj(
            conv_hidden.transpose(1, 2)
        )
        time_step, _native_b, c_readout = torch.split(
            ssm_parameters,
            [
                int(native_mixer.time_step_rank),
                int(native_mixer.ssm_state_size),
                int(native_mixer.ssm_state_size),
            ],
            dim=-1,
        )
        discrete_time_step = torch.nn.functional.softplus(
            torch.nn.functional.linear(
                time_step,
                native_mixer.dt_proj.weight,
                native_mixer.dt_proj.bias,
            )
        ).transpose(1, 2)
        a_continuous = -torch.exp(native_mixer.A_log.float())

    return {
        "residual22": residual22.detach(),
        "mixer_input": mixer_input.detach(),
        "native22": native22.detach(),
        "gate": gate.detach(),
        "c_readout": c_readout.detach(),
        "discrete_time_step": discrete_time_step.detach(),
        "a_continuous": a_continuous.detach(),
        "attention_mask": attention_mask.detach(),
    }


def _phase_b_raw_write(
    mixer_input: torch.Tensor,
    attention_mask: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> torch.Tensor:
    a_live = a_weight.to(
        device=mixer_input.device,
        dtype=mixer_input.dtype,
    )
    b_live = b_weight.to(
        device=mixer_input.device,
        dtype=mixer_input.dtype,
    )
    latent = torch.nn.functional.linear(
        mixer_input,
        a_live,
        bias=None,
    )
    raw = torch.nn.functional.linear(
        latent,
        b_live,
        bias=None,
    )
    return raw * attention_mask.to(raw.dtype).unsqueeze(-1)


def _phase_b_resume_from_raw_write(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    raw_write: torch.Tensor,
) -> torch.Tensor:
    native_mixer = wrapper.native_mixer
    shape = wrapper.correction.shape
    batch, seq_len, width = raw_write.shape
    require(width == shape.state_width, "PHASE_B_RAW_WIDTH")

    state = torch.zeros(
        (batch, shape.intermediate_size, shape.state_size),
        device=raw_write.device,
        dtype=raw_write.dtype,
    )
    outputs: list[torch.Tensor] = []
    for token_index in range(seq_len):
        discrete_a_t = torch.exp(
            context["a_continuous"][None, :, :]
            * context["discrete_time_step"][
                :, :, token_index, None
            ].float()
        ).to(dtype=raw_write.dtype)
        write_t = raw_write[:, token_index, :].reshape(
            batch,
            shape.intermediate_size,
            shape.state_size,
        )
        state = discrete_a_t * state + write_t
        read_t = torch.sum(
            state.to(context["c_readout"].dtype)
            * context["c_readout"][:, token_index, None, :],
            dim=-1,
        )
        scan_t = read_t * native_mixer.act(
            context["gate"][:, :, token_index]
        )
        outputs.append(
            torch.nn.functional.linear(
                scan_t,
                native_mixer.out_proj.weight,
                bias=None,
            )
        )
    correction22 = torch.stack(outputs, dim=1)
    hidden22 = (
        context["residual22"]
        + context["native22"]
        + correction22
    )
    hidden23 = model.mamba.layers[23](
        hidden22,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    )
    final_hidden = model.mamba.norm_f(hidden23)
    output = _phase_b_joint_forward_from_hidden(
        model,
        features,
        final_hidden,
    )
    logits = output["logits"]
    require(
        tuple(logits.shape) == (batch, 3),
        "PHASE_B_LOGIT_SHAPE",
    )
    return logits


def _phase_b_margin_vector(logits: torch.Tensor) -> torch.Tensor:
    require(logits.ndim == 2 and logits.shape[1] == 3, "PHASE_B_MARGIN_LOGITS")
    return torch.stack(
        (
            logits[:, 0] - logits[:, 1],
            logits[:, 2] - logits[:, 1],
        ),
        dim=-1,
    )


def _phase_b_centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def _phase_b_two_row_projection(
    *,
    grad_refute: torch.Tensor,
    grad_support: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    require(
        grad_refute.shape == grad_support.shape == residual.shape,
        "PHASE_B_PROJECTOR_SHAPE",
    )
    g0 = grad_refute.reshape(-1).to(torch.float64)
    g1 = grad_support.reshape(-1).to(torch.float64)
    d = residual.reshape(-1).to(torch.float64)

    gram = torch.stack(
        (
            torch.stack((torch.dot(g0, g0), torch.dot(g0, g1))),
            torch.stack((torch.dot(g1, g0), torch.dot(g1, g1))),
        )
    )
    pinv = torch.linalg.pinv(
        gram,
        rtol=PHASE_B_PROJECTOR_PINV_RTOL,
        atol=0.0,
        hermitian=True,
    )
    jd = torch.stack((torch.dot(g0, d), torch.dot(g1, d)))
    alpha = pinv @ jd
    visible64 = alpha[0] * g0 + alpha[1] * g1
    denom = float(torch.dot(d, d).item())
    numer = float(torch.dot(visible64, visible64).item())
    energy = 0.0 if denom == 0.0 else numer / denom
    gain = 0.0 if denom == 0.0 else float(
        torch.linalg.vector_norm(jd).item() / math.sqrt(denom)
    )
    return (
        visible64.reshape_as(residual).to(dtype=residual.dtype),
        energy,
        gain,
    )


def _phase_b_control_energy_and_gain(
    *,
    grad_refute: torch.Tensor,
    grad_support: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[float, float]:
    g0 = grad_refute.reshape(-1).to(torch.float64)
    g1 = grad_support.reshape(-1).to(torch.float64)
    d = residual.reshape(-1).to(torch.float64)
    gram = torch.stack(
        (
            torch.stack((torch.dot(g0, g0), torch.dot(g0, g1))),
            torch.stack((torch.dot(g1, g0), torch.dot(g1, g1))),
        )
    )
    pinv = torch.linalg.pinv(
        gram,
        rtol=PHASE_B_PROJECTOR_PINV_RTOL,
        atol=0.0,
        hermitian=True,
    )
    jd = torch.stack((torch.dot(g0, d), torch.dot(g1, d)))
    denom = float(torch.dot(d, d).item())
    if denom == 0.0:
        return 0.0, 0.0
    projected_energy = float((jd @ pinv @ jd).item()) / denom
    gain = float(torch.linalg.vector_norm(jd).item() / math.sqrt(denom))
    return projected_energy, gain


def _phase_b_effect_sums(
    source_logits: torch.Tensor,
    full_logits: torch.Tensor,
    visible_logits: torch.Tensor,
    complement_logits: torch.Tensor,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    transforms = {
        "centered_logits": _phase_b_centered_logits,
        "two_margins": _phase_b_margin_vector,
    }
    for label, transform in transforms.items():
        source = transform(source_logits).to(torch.float64)
        full = transform(full_logits).to(torch.float64) - source
        visible = transform(visible_logits).to(torch.float64) - source
        complement = transform(complement_logits).to(torch.float64) - source
        interaction = full - visible - complement
        result[label] = {
            "E_full": float(torch.sum(full * full).item()),
            "E_visible": float(torch.sum(visible * visible).item()),
            "E_complement": float(torch.sum(complement * complement).item()),
            "E_interaction": float(torch.sum(interaction * interaction).item()),
        }
    return result


def _phase_b_empty_orientation(source: str, target: str) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "count": 0,
        "residual_diff_sq": 0.0,
        "residual_source_sq": 0.0,
        "residual_target_sq": 0.0,
        "actual_task_row_energy_sum": 0.0,
        "actual_directional_gain_sum": 0.0,
        "control_task_row_energy_sums": [0.0] * PHASE_B_CONTROL_COUNT,
        "control_directional_gain_sums": [0.0] * PHASE_B_CONTROL_COUNT,
        "full_replay_max_abs": 0.0,
        "finite": {
            "centered_logits": {
                "E_full": 0.0,
                "E_visible": 0.0,
                "E_complement": 0.0,
                "E_interaction": 0.0,
            },
            "two_margins": {
                "E_full": 0.0,
                "E_visible": 0.0,
                "E_complement": 0.0,
                "E_interaction": 0.0,
            },
        },
        "visible_prediction_disagreement_vs_source": 0,
        "complement_prediction_disagreement_vs_source": 0,
    }


def _phase_b_worker_source_cells(worker_id: int) -> tuple[tuple[int, int], ...]:
    require(worker_id in (0, 1), "PHASE_B_WORKER_ID")
    ordered = tuple(FULL_FACTORIAL_CELLS)
    return tuple(
        cell
        for index, cell in enumerate(ordered)
        if index % 2 == worker_id
    )


def _phase_b_worker_orientations(
    worker_id: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rows = []
    for source in _phase_b_worker_source_cells(worker_id):
        a, r = source
        for target_a in FACTOR_SEEDS:
            if target_a != a:
                rows.append((source, (target_a, r)))
    return tuple(rows)


def _phase_b_batch_features(
    features: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in features.items()}


def _phase_b_accumulate_orientation_batch(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    raw_source: torch.Tensor,
    raw_target: torch.Tensor,
    accumulator: dict[str, Any],
) -> None:
    residual = raw_target - raw_source
    accumulator["residual_diff_sq"] += float(
        torch.sum(residual.to(torch.float64) ** 2).item()
    )
    accumulator["residual_source_sq"] += float(
        torch.sum(raw_source.to(torch.float64) ** 2).item()
    )
    accumulator["residual_target_sq"] += float(
        torch.sum(raw_target.to(torch.float64) ** 2).item()
    )

    raw_leaf = raw_source.detach().clone().requires_grad_(True)
    source_logits = _phase_b_resume_from_raw_write(
        model=model,
        wrapper=wrapper,
        features=features,
        context=context,
        raw_write=raw_leaf,
    )
    margins = _phase_b_margin_vector(source_logits)
    grad_refute = torch.autograd.grad(
        margins[:, 0].sum(),
        raw_leaf,
        retain_graph=True,
        create_graph=False,
    )[0]
    grad_support = torch.autograd.grad(
        margins[:, 1].sum(),
        raw_leaf,
        retain_graph=False,
        create_graph=False,
    )[0]

    visible_rows = []
    for row_index in range(int(residual.shape[0])):
        visible, energy, gain = _phase_b_two_row_projection(
            grad_refute=grad_refute[row_index],
            grad_support=grad_support[row_index],
            residual=residual[row_index],
        )
        visible_rows.append(visible)
        accumulator["actual_task_row_energy_sum"] += energy
        accumulator["actual_directional_gain_sum"] += gain

        for control_index in range(PHASE_B_CONTROL_COUNT):
            controlled = phase_b_apply_signed_permutation(
                residual[row_index : row_index + 1],
                control_index=control_index,
            )[0]
            control_energy, control_gain = _phase_b_control_energy_and_gain(
                grad_refute=grad_refute[row_index],
                grad_support=grad_support[row_index],
                residual=controlled,
            )
            accumulator["control_task_row_energy_sums"][
                control_index
            ] += control_energy
            accumulator["control_directional_gain_sums"][
                control_index
            ] += control_gain

    visible = torch.stack(visible_rows, dim=0)
    complement = residual - visible

    with torch.no_grad():
        target_logits = _phase_b_resume_from_raw_write(
            model=model,
            wrapper=wrapper,
            features=features,
            context=context,
            raw_write=raw_target,
        )
        full_logits = _phase_b_resume_from_raw_write(
            model=model,
            wrapper=wrapper,
            features=features,
            context=context,
            raw_write=raw_source + residual,
        )
        visible_logits = _phase_b_resume_from_raw_write(
            model=model,
            wrapper=wrapper,
            features=features,
            context=context,
            raw_write=raw_source + visible,
        )
        complement_logits = _phase_b_resume_from_raw_write(
            model=model,
            wrapper=wrapper,
            features=features,
            context=context,
            raw_write=raw_source + complement,
        )

    replay_error = float(
        torch.max(torch.abs(full_logits - target_logits)).item()
    )
    accumulator["full_replay_max_abs"] = max(
        accumulator["full_replay_max_abs"],
        replay_error,
    )

    effects = _phase_b_effect_sums(
        source_logits.detach(),
        full_logits,
        visible_logits,
        complement_logits,
    )
    for coordinate in ("centered_logits", "two_margins"):
        for key in (
            "E_full",
            "E_visible",
            "E_complement",
            "E_interaction",
        ):
            accumulator["finite"][coordinate][key] += effects[coordinate][key]

    source_pred = torch.argmax(source_logits.detach(), dim=-1)
    visible_pred = torch.argmax(visible_logits, dim=-1)
    complement_pred = torch.argmax(complement_logits, dim=-1)
    accumulator["visible_prediction_disagreement_vs_source"] += int(
        torch.count_nonzero(source_pred != visible_pred).item()
    )
    accumulator["complement_prediction_disagreement_vs_source"] += int(
        torch.count_nonzero(source_pred != complement_pred).item()
    )
    accumulator["count"] += int(residual.shape[0])


def _phase_b_finalize_orientation(value: Mapping[str, Any]) -> dict[str, Any]:
    count = int(value["count"])
    require(count == DEV_ROWS, f"PHASE_B_ORIENTATION_COUNT:{count}")
    denom_sq = 0.5 * (
        float(value["residual_source_sq"])
        + float(value["residual_target_sq"])
    )
    normalized = (
        None
        if denom_sq <= 0.0
        else math.sqrt(float(value["residual_diff_sq"]) / denom_sq)
    )
    return {
        **dict(value),
        "normalized_residual": normalized,
        "actual_task_row_energy_mean": (
            float(value["actual_task_row_energy_sum"]) / count
        ),
        "actual_directional_gain_mean": (
            float(value["actual_directional_gain_sum"]) / count
        ),
        "control_task_row_energy_means": [
            float(v) / count
            for v in value["control_task_row_energy_sums"]
        ],
        "control_directional_gain_means": [
            float(v) / count
            for v in value["control_directional_gain_sums"]
        ],
    }


def _phase_b_merge_functional_workers(
    workers: Sequence[Mapping[str, Any]],
    *,
    t: int,
) -> dict[str, Any]:
    orientations = []
    for worker in workers:
        require(int(worker["t"]) == t, "PHASE_B_WORKER_T_MISMATCH")
        orientations.extend(worker["orientations"])
    require(len(orientations) == 18, "PHASE_B_ORIENTATION_COUNT_TOTAL")
    names = {(row["source"], row["target"]) for row in orientations}
    require(len(names) == 18, "PHASE_B_ORIENTATION_DUPLICATE")

    count = sum(int(row["count"]) for row in orientations)
    require(count == 18 * DEV_ROWS, "PHASE_B_EXAMPLE_ORIENTATION_COUNT")
    normalized = [
        float(row["normalized_residual"])
        for row in orientations
        if row["normalized_residual"] is not None
    ]
    require(len(normalized) == 18, "PHASE_B_NORMALIZED_COUNT")

    actual_energy = sum(
        float(row["actual_task_row_energy_sum"])
        for row in orientations
    ) / count
    actual_gain = sum(
        float(row["actual_directional_gain_sum"])
        for row in orientations
    ) / count

    control_energy_means = []
    control_gain_means = []
    for control_index in range(PHASE_B_CONTROL_COUNT):
        control_energy_means.append(
            sum(
                float(row["control_task_row_energy_sums"][control_index])
                for row in orientations
            )
            / count
        )
        control_gain_means.append(
            sum(
                float(row["control_directional_gain_sums"][control_index])
                for row in orientations
            )
            / count
        )
    control_energy_mean = sum(control_energy_means) / PHASE_B_CONTROL_COUNT
    control_gain_mean = sum(control_gain_means) / PHASE_B_CONTROL_COUNT
    enrichment = (
        math.inf
        if control_energy_mean == 0.0 and actual_energy > 0.0
        else (
            0.0
            if control_energy_mean == 0.0
            else actual_energy / control_energy_mean
        )
    )

    finite: dict[str, Any] = {}
    for coordinate in ("centered_logits", "two_margins"):
        sums = {
            key: sum(
                float(row["finite"][coordinate][key])
                for row in orientations
            )
            for key in (
                "E_full",
                "E_visible",
                "E_complement",
                "E_interaction",
            )
        }
        e_full = sums["E_full"]
        finite[coordinate] = {
            **sums,
            "R_visible": None if e_full == 0.0 else sums["E_visible"] / e_full,
            "R_complement": None if e_full == 0.0 else sums["E_complement"] / e_full,
            "R_interaction": None if e_full == 0.0 else sums["E_interaction"] / e_full,
        }

    return {
        "t": t,
        "orientation_count": 18,
        "example_orientation_count": count,
        "normalized_residual_mean": sum(normalized) / len(normalized),
        "actual_task_row_energy_mean": actual_energy,
        "control_task_row_energy_means": control_energy_means,
        "control_task_row_energy_mean": control_energy_mean,
        "task_row_energy_enrichment": enrichment,
        "actual_directional_gain_mean": actual_gain,
        "control_directional_gain_means": control_gain_means,
        "control_directional_gain_mean": control_gain_mean,
        "full_replay_max_abs": max(
            float(row["full_replay_max_abs"])
            for row in orientations
        ),
        "finite_intervention": finite,
        "visible_prediction_disagreement_vs_source": sum(
            int(row["visible_prediction_disagreement_vs_source"])
            for row in orientations
        ),
        "complement_prediction_disagreement_vs_source": sum(
            int(row["complement_prediction_disagreement_vs_source"])
            for row in orientations
        ),
        "orientations": orientations,
    }


def _phase_b_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    centered = value["finite_intervention"]["centered_logits"]
    margins = value["finite_intervention"]["two_margins"]

    def coordinate_pass(row: Mapping[str, Any]) -> bool:
        return (
            row["R_visible"] is not None
            and 0.60 <= float(row["R_visible"]) <= 1.40
            and row["R_complement"] is not None
            and float(row["R_complement"]) <= 0.05
            and row["R_interaction"] is not None
            and float(row["R_interaction"]) <= 0.05
        )

    centered_pass = coordinate_pass(centered)
    margins_pass = coordinate_pass(margins)
    enrichment_pass = (
        float(value["task_row_energy_enrichment"]) >= 5.0
    )
    replay_pass = float(value["full_replay_max_abs"]) <= PHASE_B_FULL_REPLAY_ATOL
    return {
        "centered_logits_pass": centered_pass,
        "two_margins_pass": margins_pass,
        "enrichment_pass": enrichment_pass,
        "full_replay_pass": replay_pass,
        "pass": (
            centered_pass
            and margins_pass
            and enrichment_pass
            and replay_pass
        ),
    }


def _phase_b_endpoint_authentication(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    require(int(value["t"]) == 20, "PHASE_B_ENDPOINT_T")
    checks = {}

    def check(name: str, observed: float, target: float, atol: float) -> None:
        checks[name] = {
            "observed": observed,
            "target": target,
            "atol": atol,
            "abs_error": abs(observed - target),
            "pass": abs(observed - target) <= atol,
        }

    check(
        "normalized_residual",
        float(value["normalized_residual_mean"]),
        PHASE_B_ENDPOINT_TARGET["normalized_residual"],
        PHASE_B_ENDPOINT_TOLERANCE["normalized_residual_atol"],
    )
    check(
        "task_row_energy",
        float(value["actual_task_row_energy_mean"]),
        PHASE_B_ENDPOINT_TARGET["task_row_energy"],
        PHASE_B_ENDPOINT_TOLERANCE["task_row_energy_atol"],
    )
    check(
        "control_task_row_energy",
        float(value["control_task_row_energy_mean"]),
        PHASE_B_ENDPOINT_TARGET["control_task_row_energy"],
        PHASE_B_ENDPOINT_TOLERANCE["control_task_row_energy_atol"],
    )
    check(
        "enrichment",
        float(value["task_row_energy_enrichment"]),
        PHASE_B_ENDPOINT_TARGET["enrichment"],
        PHASE_B_ENDPOINT_TOLERANCE["enrichment_atol"],
    )
    for coordinate in ("centered_logits", "two_margins"):
        observed_row = value["finite_intervention"][coordinate]
        target_row = PHASE_B_ENDPOINT_TARGET[coordinate]
        for metric in ("R_visible", "R_complement", "R_interaction"):
            check(
                f"{coordinate}.{metric}",
                float(observed_row[metric]),
                float(target_row[metric]),
                PHASE_B_ENDPOINT_TOLERANCE["effect_ratio_atol"],
            )
    checks["full_replay"] = {
        "observed": float(value["full_replay_max_abs"]),
        "target_max": PHASE_B_FULL_REPLAY_ATOL,
        "pass": float(value["full_replay_max_abs"]) <= PHASE_B_FULL_REPLAY_ATOL,
    }
    passed = all(bool(row["pass"]) for row in checks.values())
    return {"pass": passed, "checks": checks}


def _phase_b_interpretation(
    *,
    geometric_birth: int,
    functional_birth: int | None,
) -> str:
    if geometric_birth == 1 and functional_birth == 1:
        return "IMMEDIATE_FIRST_UPDATE_BIRTH"
    if geometric_birth == 1 and functional_birth is not None:
        return "GEOMETRY_FIRST_FUNCTION_LATER"
    if geometric_birth > 1:
        return "DELAYED_GEOMETRIC_BIRTH"
    return "NO_FUNCTIONAL_FREEDOM_BIRTH_LOCALIZED"


def _phase_b_prepare_worker_runtime(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    torch.nn.Module,
    Any,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    summary, trajectory = _phase_b_load_phase_a_artifacts()
    static, encoded, snapshot, checkpoint_path = _prepare_runtime(args)
    del static
    model, wrapper, _runtime_meta, strong_mask, planes = base._prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        a_init_seed=6201,
        training_rng_seed=6201,
    )
    model.eval()
    model.mamba.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return (
        summary,
        trajectory,
        encoded,
        model,
        wrapper,
        strong_mask,
        planes,
    )


def run_phase_b_worker(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=False,
    )
    validate_runtime_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    validate_phase_b_static_contract()
    require(torch.cuda.is_available(), "PHASE_B_WORKER_CUDA")
    require(torch.cuda.device_count() == 1, "PHASE_B_WORKER_VISIBLE_GPU_COUNT")
    require(args.worker_id in (0, 1), "PHASE_B_WORKER_ID")
    require(args.phase_b_step is not None, "PHASE_B_WORKER_STEP")
    require(args.scratch_root is not None, "PHASE_B_WORKER_SCRATCH")

    (
        _phase_a_summary,
        trajectory,
        encoded,
        model,
        wrapper,
        strong_mask,
        planes,
    ) = _phase_b_prepare_worker_runtime(args)

    device = torch.device("cuda:0")
    features, _labels, active, targets = p3a._feature_batch_to_device(
        encoded["dev_bundle"],
        device,
    )
    orientations = _phase_b_worker_orientations(args.worker_id)
    accumulators = {
        (cell_name(*source), cell_name(*target)): _phase_b_empty_orientation(
            cell_name(*source),
            cell_name(*target),
        )
        for source, target in orientations
    }
    covariance = torch.zeros((768, 768), dtype=torch.float64)
    valid_token_count = 0

    t = int(args.phase_b_step)
    snapshot_weights = {
        cell: _phase_b_snapshot(trajectory, cell, t)
        for cell in FULL_FACTORIAL_CELLS
    }

    for start in range(0, DEV_ROWS, PHASE_B_FUNCTIONAL_BATCH_ROWS):
        stop = min(start + PHASE_B_FUNCTIONAL_BATCH_ROWS, DEV_ROWS)
        batch_features = _phase_b_batch_features(features, start, stop)
        batch_active = active[start:stop]
        batch_targets = targets[start:stop]
        context = _phase_b_prepare_common_context(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            stressor_active=batch_active,
            target_indices=batch_targets,
            strong_mask=strong_mask,
            planes=planes,
        )

        if args.phase_b_compute_geometry:
            valid = batch_features["attention_mask"].bool()
            x = context["mixer_input"][valid].detach().cpu().to(torch.float64)
            covariance += x.T @ x
            valid_token_count += int(x.shape[0])

        raw_by_cell = {
            cell: _phase_b_raw_write(
                context["mixer_input"],
                batch_features["attention_mask"],
                snapshot_weights[cell][0],
                snapshot_weights[cell][1],
            ).detach()
            for cell in FULL_FACTORIAL_CELLS
        }

        for source, target in orientations:
            accumulator = accumulators[
                (cell_name(*source), cell_name(*target))
            ]
            _phase_b_accumulate_orientation_batch(
                model=model,
                wrapper=wrapper,
                features=batch_features,
                context=context,
                raw_source=raw_by_cell[source],
                raw_target=raw_by_cell[target],
                accumulator=accumulator,
            )

    finalized = [
        _phase_b_finalize_orientation(value)
        for value in accumulators.values()
    ]
    result: dict[str, Any] = {
        "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_WORKER_V1",
        "worker_id": args.worker_id,
        "t": t,
        "source_cells": [
            cell_name(*cell)
            for cell in _phase_b_worker_source_cells(args.worker_id)
        ],
        "orientations": finalized,
        "backward_executed": False,
        "parameter_gradients_accumulated": any(
            parameter.grad is not None
            for parameter in model.parameters()
        ),
        "optimizer_constructed": False,
        "training_executed": False,
        "confirmatory_9601_9900_loaded": False,
    }
    require(
        result["parameter_gradients_accumulated"] is False,
        "PHASE_B_PARAMETER_GRADIENT_ACCUMULATED",
    )

    if args.phase_b_compute_geometry:
        require(
            valid_token_count == PHASE_B_VALID_TOKEN_COUNT,
            f"PHASE_B_VALID_TOKEN_COUNT:{valid_token_count}",
        )
        result["valid_token_count"] = valid_token_count
        result["geometry"] = phase_b_geometry_trajectory(
            trajectory,
            covariance,
        )

    worker_root = Path(args.scratch_root) / f"worker{args.worker_id}"
    worker_root.mkdir(parents=True, exist_ok=False)
    (worker_root / "worker_result.json").write_bytes(
        canonical_json_bytes(result)
    )
    print(
        "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_WORKER_PASS "
        f"worker={args.worker_id} t={t} "
        f"orientations={len(finalized)}"
    )


def _phase_b_spawn_workers(
    args: argparse.Namespace,
    *,
    t: int,
    scratch_root: Path,
    compute_geometry: bool,
    log_prefix: str,
) -> tuple[list[dict[str, Any]], list[Path]]:
    processes = []
    logs: list[Path] = []
    script = Path(__file__).resolve()
    worker_scratch = scratch_root / f"{log_prefix}_scratch"
    worker_scratch.mkdir(parents=True, exist_ok=False)
    for worker_id in (0, 1):
        log_path = scratch_root / f"{log_prefix}_worker{worker_id}.log"
        logs.append(log_path)
        command = [
            sys.executable,
            str(script),
            "--phase-b-worker",
            "--expected-head",
            args.expected_head,
            "--implementation-freeze-commit",
            args.implementation_freeze_commit,
            "--model-snapshot",
            str(args.model_snapshot),
            "--tokenizer-snapshot",
            str(args.tokenizer_snapshot),
            "--checkpoint",
            str(args.checkpoint),
            "--scratch-root",
            str(worker_scratch),
            "--worker-id",
            str(worker_id),
            "--phase-b-step",
            str(t),
        ]
        if compute_geometry and worker_id == 0:
            command.append("--phase-b-compute-geometry")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, handle))

    return_codes = []
    for process, handle in processes:
        return_codes.append(process.wait())
        handle.close()
    if any(code != 0 for code in return_codes):
        for worker_id, log_path in enumerate(logs):
            print(f"=== PHASE B WORKER {worker_id} LOG ===")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
        raise TemporalBirthError(
            f"PHASE_B_WORKER_FAILURE:t={t}:codes={return_codes}"
        )

    workers = []
    for worker_id in (0, 1):
        result_path = worker_scratch / f"worker{worker_id}" / "worker_result.json"
        require(result_path.is_file(), "PHASE_B_WORKER_RESULT_MISSING")
        workers.append(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
    return workers, logs


def run_phase_b_cuda_preflight(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=False,
    )
    validate_runtime_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    validate_phase_b_static_contract()
    _validate_two_t4s()

    (
        _summary,
        trajectory,
        encoded,
        model,
        wrapper,
        strong_mask,
        planes,
    ) = _phase_b_prepare_worker_runtime(args)
    device = torch.device("cuda:0")
    features, _labels, active, targets = p3a._feature_batch_to_device(
        encoded["dev_bundle"],
        device,
    )
    batch_features = _phase_b_batch_features(features, 0, 1)
    context = _phase_b_prepare_common_context(
        model=model,
        wrapper=wrapper,
        features=batch_features,
        stressor_active=active[:1],
        target_indices=targets[:1],
        strong_mask=strong_mask,
        planes=planes,
    )

    source = (6201, 6201)
    target = (6202, 6201)
    a0, b0 = _phase_b_snapshot(trajectory, source, 0)
    raw0 = _phase_b_raw_write(
        context["mixer_input"],
        batch_features["attention_mask"],
        a0,
        b0,
    )
    require(
        int(torch.count_nonzero(raw0).item()) == 0,
        "PHASE_B_PREFLIGHT_T0_RAW_NONZERO",
    )

    a_s, b_s = _phase_b_snapshot(trajectory, source, 20)
    a_t, b_t = _phase_b_snapshot(trajectory, target, 20)
    raw_source = _phase_b_raw_write(
        context["mixer_input"],
        batch_features["attention_mask"],
        a_s,
        b_s,
    ).detach()
    raw_target = _phase_b_raw_write(
        context["mixer_input"],
        batch_features["attention_mask"],
        a_t,
        b_t,
    ).detach()
    leaf = raw_source.clone().requires_grad_(True)
    logits = _phase_b_resume_from_raw_write(
        model=model,
        wrapper=wrapper,
        features=batch_features,
        context=context,
        raw_write=leaf,
    )
    margins = _phase_b_margin_vector(logits)
    g0 = torch.autograd.grad(
        margins[:, 0].sum(),
        leaf,
        retain_graph=True,
    )[0]
    g1 = torch.autograd.grad(
        margins[:, 1].sum(),
        leaf,
    )[0]
    require(bool(torch.isfinite(g0).all()), "PHASE_B_PREFLIGHT_G0_NONFINITE")
    require(bool(torch.isfinite(g1).all()), "PHASE_B_PREFLIGHT_G1_NONFINITE")

    with torch.no_grad():
        target_logits = _phase_b_resume_from_raw_write(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            context=context,
            raw_write=raw_target,
        )
        full_logits = _phase_b_resume_from_raw_write(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            context=context,
            raw_write=raw_source + (raw_target - raw_source),
        )
    full_error = float(
        torch.max(torch.abs(full_logits - target_logits)).item()
    )
    require(
        full_error <= PHASE_B_FULL_REPLAY_ATOL,
        f"PHASE_B_PREFLIGHT_FULL_REPLAY:{full_error}",
    )
    require(
        not any(parameter.grad is not None for parameter in model.parameters()),
        "PHASE_B_PREFLIGHT_PARAMETER_GRAD",
    )

    print("GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_CUDA_PREFLIGHT_PASS")
    print("T0_RAW_WRITE_EXACT_ZERO=True")
    print("JOINT_AUTOGRAD_ROWS_FINITE=True")
    print(f"FULL_REPLAY_MAX_ABS={full_error:.17g}")
    print("BACKWARD_EXECUTED=False")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("TRAINING_EXECUTED=False")
    print("PREFLIGHT_COLLECTION=FORBIDDEN")
    print("CONFIRMATORY_9601_9900_LOADED=False")


def run_phase_b_analysis(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=False,
    )
    validate_runtime_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
    )
    validate_phase_b_static_contract()
    gpu_meta = _validate_two_t4s()
    require(args.output_root is not None, "PHASE_B_OUTPUT_REQUIRED")
    output_root = Path(args.output_root)
    require(not output_root.exists(), f"PHASE_B_OUTPUT_COLLISION:{output_root}")

    phase_a_summary, _trajectory = _phase_b_load_phase_a_artifacts()
    scratch_root = Path(
        tempfile.mkdtemp(
            prefix="gen5_temporal_birth_phase_b_",
            dir="/kaggle/working",
        )
    )

    all_logs: list[Path] = []
    endpoint_workers, endpoint_logs = _phase_b_spawn_workers(
        args,
        t=20,
        scratch_root=scratch_root,
        compute_geometry=True,
        log_prefix="endpoint_t20",
    )
    all_logs.extend(endpoint_logs)
    endpoint = _phase_b_merge_functional_workers(
        endpoint_workers,
        t=20,
    )
    endpoint_auth = _phase_b_endpoint_authentication(endpoint)
    require(
        endpoint_auth["pass"] is True,
        "PHASE_B_STEP20_AUTHENTICATION_FAILED",
    )

    geometry = endpoint_workers[0].get("geometry")
    require(isinstance(geometry, list) and len(geometry) == 21, "PHASE_B_GEOMETRY")
    geometric_birth = phase_b_geometric_birth_step(geometry)

    functional_scan: list[dict[str, Any]] = []
    functional_birth: int | None = None
    endpoint_gate = _phase_b_gate(endpoint)
    for t in range(geometric_birth, TOTAL_OPTIMIZER_STEPS + 1):
        if t == 20:
            value = endpoint
            gate = endpoint_gate
        else:
            workers, logs = _phase_b_spawn_workers(
                args,
                t=t,
                scratch_root=scratch_root,
                compute_geometry=False,
                log_prefix=f"scan_t{t:02d}",
            )
            all_logs.extend(logs)
            value = _phase_b_merge_functional_workers(workers, t=t)
            gate = _phase_b_gate(value)
        functional_scan.append(
            {
                "t": t,
                "metrics": value,
                "gate": gate,
            }
        )
        if gate["pass"]:
            functional_birth = t
            break

    interpretation = _phase_b_interpretation(
        geometric_birth=geometric_birth,
        functional_birth=functional_birth,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    log_root = output_root / "worker_logs"
    log_root.mkdir()
    for index, log_path in enumerate(all_logs):
        shutil.copy2(log_path, log_root / f"{index:03d}_{log_path.name}")

    metrics_path = output_root / "temporal_birth_metrics.pt"
    torch.save(
        {
            "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_METRICS_V1",
            "execution_head": args.expected_head,
            "implementation_freeze_commit": args.implementation_freeze_commit,
            "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
            "geometry": geometry,
            "endpoint": endpoint,
            "endpoint_authentication": endpoint_auth,
            "functional_scan": functional_scan,
            "control_identity": [
                phase_b_control_identity(index)
                for index in range(PHASE_B_CONTROL_COUNT)
            ],
        },
        metrics_path,
    )

    summary = {
        "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_SUMMARY_V1",
        "result": interpretation,
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "phase_a_evidence_freeze_commit": PHASE_A_EVIDENCE_FREEZE_COMMIT,
        "phase_a_summary_sha256": PHASE_A_SUMMARY_SHA256,
        "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
        "pressure": PRESSURE,
        "arm": ARM,
        "dev_rows": DEV_ROWS,
        "valid_token_count": endpoint_workers[0]["valid_token_count"],
        "geometric_birth_step": geometric_birth,
        "functional_freedom_birth_step": functional_birth,
        "endpoint_authentication": endpoint_auth,
        "endpoint_gate": endpoint_gate,
        "functional_steps_evaluated": [
            int(row["t"]) for row in functional_scan
        ],
        "fixed_gate": {
            "R_visible": [0.60, 1.40],
            "R_complement_max": 0.05,
            "R_interaction_max": 0.05,
            "task_row_energy_enrichment_min": 5.0,
        },
        "projector_pinv_rtol": PHASE_B_PROJECTOR_PINV_RTOL,
        "functional_batch_rows": PHASE_B_FUNCTIONAL_BATCH_ROWS,
        "control_family": [
            phase_b_control_identity(index)
            for index in range(PHASE_B_CONTROL_COUNT)
        ],
        "gpu_topology": GPU_TOPOLOGY,
        "gpu_runtime": gpu_meta,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "parameter_gradients_accumulated": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
    }
    summary_path = output_root / "temporal_birth_analysis_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary))

    provenance = {
        "schema_version": "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_PROVENANCE_V1",
        "status": "PASS",
        "authority_commit": AUTHORITY_COMMIT,
        "authority_blob": git("rev-parse", f"HEAD:{AUTHORITY_PATH}"),
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "phase_a_evidence_freeze_commit": PHASE_A_EVIDENCE_FREEZE_COMMIT,
        "phase_a_summary_sha256": PHASE_A_SUMMARY_SHA256,
        "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
        "internal_precursor_metrics_sha256": INTERNAL_PRECURSOR_METRICS_SHA256,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "frozen_snapshot_revision": FROZEN_SNAPSHOT_REVISION,
        "dev_encoding_sha256": phase_a_summary["dev_encoding_sha256"],
        "gpu_topology": GPU_TOPOLOGY,
        "summary_sha256": sha256_file(summary_path),
        "metrics_sha256": sha256_file(metrics_path),
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "parameter_gradients_accumulated": False,
        "confirmatory_9601_9900_loaded": False,
        "failed_run_collection_allowed": False,
        "preflight_collection_allowed": False,
    }
    provenance_path = output_root / "run_provenance.json"
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    print("GEN5_AINIT_TEMPORAL_BIRTH_PHASE_B_PASS")
    print(f"RESULT={interpretation}")
    print(f"GEOMETRIC_BIRTH_STEP={geometric_birth}")
    print(
        "FUNCTIONAL_FREEDOM_BIRTH_STEP="
        + (
            "NONE"
            if functional_birth is None
            else str(functional_birth)
        )
    )
    print("STEP20_AUTHENTICATION_PASS=True")
    print("BACKWARD_EXECUTED=False")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={metrics_path}")
    print(f"PROVENANCE={provenance_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--cuda-preflight-only", action="store_true")
    modes.add_argument("--replay-auth-recovery-only", action="store_true")
    modes.add_argument("--replay-auth-recovery-worker", action="store_true")
    modes.add_argument("--run-replay-matrix", action="store_true")
    modes.add_argument("--run-replay-worker", action="store_true")
    modes.add_argument("--phase-b-cuda-preflight-only", action="store_true")
    modes.add_argument("--run-phase-b", action="store_true")
    modes.add_argument("--phase-b-worker", action="store_true")

    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--allow-opening-worktree", action="store_true")
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--phase-b-step", type=int)
    parser.add_argument("--phase-b-compute-geometry", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    runtime_fields = (
        "implementation_freeze_commit",
        "model_snapshot",
        "tokenizer_snapshot",
        "checkpoint",
        "output_root",
        "scratch_root",
        "worker_id",
        "phase_b_step",
    )
    if args.static_verify_only:
        for field in runtime_fields:
            require(getattr(args, field) is None, f"STATIC_RUNTIME_ARG:{field}")
        require(not args.phase_b_compute_geometry, "STATIC_PHASE_B_GEOMETRY_FORBIDDEN")
        return

    require(not args.allow_opening_worktree, "RUNTIME_OPENING_WORKTREE_FORBIDDEN")
    require(
        args.implementation_freeze_commit is not None,
        "IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")

    if args.cuda_preflight_only:
        require(args.output_root is None, "PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.scratch_root is None, "PREFLIGHT_SCRATCH_FORBIDDEN")
        require(args.worker_id is None, "PREFLIGHT_WORKER_FORBIDDEN")
        require(args.phase_b_step is None, "PREFLIGHT_PHASE_B_STEP_FORBIDDEN")
        require(not args.phase_b_compute_geometry, "PREFLIGHT_PHASE_B_GEOMETRY_FORBIDDEN")
    elif args.phase_b_cuda_preflight_only:
        require(args.output_root is None, "PHASE_B_PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.scratch_root is None, "PHASE_B_PREFLIGHT_SCRATCH_FORBIDDEN")
        require(args.worker_id is None, "PHASE_B_PREFLIGHT_WORKER_FORBIDDEN")
        require(args.phase_b_step is None, "PHASE_B_PREFLIGHT_STEP_FORBIDDEN")
        require(not args.phase_b_compute_geometry, "PHASE_B_PREFLIGHT_GEOMETRY_FORBIDDEN")
    elif args.run_phase_b:
        require(args.output_root is not None, "PHASE_B_OUTPUT_REQUIRED")
        require(args.scratch_root is None, "PHASE_B_MAIN_SCRATCH_FORBIDDEN")
        require(args.worker_id is None, "PHASE_B_MAIN_WORKER_FORBIDDEN")
        require(args.phase_b_step is None, "PHASE_B_MAIN_STEP_FORBIDDEN")
        require(not args.phase_b_compute_geometry, "PHASE_B_MAIN_GEOMETRY_FORBIDDEN")
    elif args.phase_b_worker:
        require(args.output_root is None, "PHASE_B_WORKER_OUTPUT_FORBIDDEN")
        require(args.scratch_root is not None, "PHASE_B_WORKER_SCRATCH_REQUIRED")
        require(args.worker_id in (0, 1), "PHASE_B_WORKER_ID_REQUIRED")
        require(args.phase_b_step is not None and 0 <= args.phase_b_step <= TOTAL_OPTIMIZER_STEPS, "PHASE_B_WORKER_STEP_REQUIRED")
        if args.phase_b_compute_geometry:
            require(args.worker_id == 0, "PHASE_B_GEOMETRY_WORKER0_ONLY")
    elif args.replay_auth_recovery_only:
        require(args.output_root is None, "RECOVERY_OUTPUT_FORBIDDEN")
        require(args.scratch_root is None, "RECOVERY_SCRATCH_FORBIDDEN")
        require(args.worker_id is None, "RECOVERY_WORKER_FORBIDDEN")
    elif args.replay_auth_recovery_worker:
        require(args.output_root is None, "RECOVERY_WORKER_OUTPUT_FORBIDDEN")
        require(args.scratch_root is None, "RECOVERY_WORKER_SCRATCH_FORBIDDEN")
        require(args.worker_id in (0, 1), "RECOVERY_WORKER_ID_REQUIRED")
    elif args.run_replay_matrix:
        require(args.output_root is not None, "MATRIX_OUTPUT_REQUIRED")
        require(args.scratch_root is None, "MATRIX_SCRATCH_FORBIDDEN")
        require(args.worker_id is None, "MATRIX_WORKER_FORBIDDEN")
    else:
        require(args.output_root is None, "WORKER_OUTPUT_FORBIDDEN")
        require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.static_verify_only:
        run_static_verify(args)
    elif args.cuda_preflight_only:
        run_cuda_preflight(args)
    elif args.replay_auth_recovery_only:
        run_replay_auth_recovery(args)
    elif args.replay_auth_recovery_worker:
        run_replay_auth_recovery_worker(args)
    elif args.run_replay_matrix:
        run_replay_matrix(args)
    elif args.phase_b_cuda_preflight_only:
        run_phase_b_cuda_preflight(args)
    elif args.run_phase_b:
        run_phase_b_analysis(args)
    elif args.phase_b_worker:
        run_phase_b_worker(args)
    else:
        run_replay_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
