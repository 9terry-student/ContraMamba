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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--cuda-preflight-only", action="store_true")
    modes.add_argument("--replay-auth-recovery-only", action="store_true")
    modes.add_argument("--replay-auth-recovery-worker", action="store_true")
    modes.add_argument("--run-replay-matrix", action="store_true")
    modes.add_argument("--run-replay-worker", action="store_true")

    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--allow-opening-worktree", action="store_true")
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--worker-id", type=int)
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
    )
    if args.static_verify_only:
        for field in runtime_fields:
            require(getattr(args, field) is None, f"STATIC_RUNTIME_ARG:{field}")
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
    else:
        run_replay_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
