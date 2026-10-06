#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import audit_reason_router_gen5_ainit_temporal_birth as tb  # noqa: E402


ROOT = tb.ROOT
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
AUTHORITY_COMMIT = "e05948ffacab4210158fbd87eeac7bb25b733556"
AUTHORITY_PATH = (
    "reports/reason_router_gen5_ainit_t1_factor_swap_"
    "causal_authority_spec_candidate.md"
)
SOURCE_HEAD = "4841ed1d106f625e690c9cd1ae021ab6040f7afc"

PHASE_A_TRAJECTORY_PATH = Path(tb.PHASE_A_TRAJECTORY_PATH)
PHASE_A_TRAJECTORY_SHA256 = tb.PHASE_A_TRAJECTORY_SHA256

BEHAVIORAL_COORDINATES_PATH = Path(
    "reports/reason_router_gen5_ainit_temporal_mechanism_recovery_runs/"
    "gen5-ainit-temporal-mechanism-d9b790b-r1-partial-recovery/"
    "temporal_behavioral_coordinates.pt"
)
BEHAVIORAL_COORDINATES_SHA256 = (
    "41518c0dbac345b972bf61920fe98681541f6393d6770494acb0df1cce149101"
)
SYNTHESIS_PATH = Path(
    "reports/reason_router_gen5_ainit_temporal_mechanism_"
    "static_synthesis_report_candidate.md"
)
SYNTHESIS_SHA256 = (
    "a9aee61287c09bc4181fc79d6805319996706e52f09cb2cb25d0d5b298e6266b"
)
SHARED_VULNERABILITY_PATH = Path(tb.TEMPORAL_MECHANISM_SHARED_VULNERABILITY_PATH)

FACTOR_SEEDS = tuple(tb.FACTOR_SEEDS)
DEV_ROWS = tb.DEV_ROWS
ROW_BATCH_SIZE = 32
AUTH_ROW_START = 0
AUTH_ROW_STOP = 32
WORKER_RANGES = {0: (0, 420), 1: (420, 840)}
ANCHOR_AUTH_ATOL = 5.0e-5
AFFINITY_EPS = 1.0e-12

IMPLEMENTATION_PATHS = {
    "scripts/audit_reason_router_gen5_ainit_t1_factor_swap.py",
    "tests/test_reason_router_gen5_ainit_t1_factor_swap.py",
}


class FactorSwapError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FactorSwapError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_bytes(data)
    os.replace(temp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temp)
    os.replace(temp, path)


def status_paths() -> set[str]:
    return tb.status_paths()


def hybrid_name(a_rec: int, a_don: int, r_don: int) -> str:
    return f"AREC{a_rec}-ADON{a_don}-R{r_don}"


def all_hybrids() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (a_rec, a_don, r_don)
        for a_rec in FACTOR_SEEDS
        for a_don in FACTOR_SEEDS
        for r_don in FACTOR_SEEDS
    )


def cross_hybrids() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        state for state in all_hybrids()
        if state[0] != state[1]
    )


ALL_HYBRIDS = all_hybrids()
CROSS_HYBRIDS = cross_hybrids()
require(len(ALL_HYBRIDS) == 27, "FACTOR_SWAP_ALL_STATE_COUNT")
require(len(CROSS_HYBRIDS) == 18, "FACTOR_SWAP_CROSS_STATE_COUNT")


def worker_batches(worker_id: int) -> tuple[tuple[int, int], ...]:
    require(worker_id in WORKER_RANGES, f"FACTOR_SWAP_WORKER_ID:{worker_id}")
    start, stop = WORKER_RANGES[worker_id]
    rows = []
    for left in range(start, stop, ROW_BATCH_SIZE):
        rows.append((left, min(left + ROW_BATCH_SIZE, stop)))
    return tuple(rows)


def planned_counts() -> dict[str, Any]:
    return {
        "row_batch_size": ROW_BATCH_SIZE,
        "worker_rows": {
            str(k): list(v) for k, v in WORKER_RANGES.items()
        },
        "batches_per_worker": {
            str(k): len(worker_batches(k)) for k in (0, 1)
        },
        "all_states": len(ALL_HYBRIDS),
        "new_cross_states": len(CROSS_HYBRIDS),
        "full_matched_anchor_recompute": 0,
        "cross_forward_calls_per_worker": {
            str(k): len(worker_batches(k)) * len(CROSS_HYBRIDS)
            for k in (0, 1)
        },
        "cross_forward_calls_total": sum(
            len(worker_batches(k)) * len(CROSS_HYBRIDS)
            for k in (0, 1)
        ),
        "matched_anchor_auth_rows": [AUTH_ROW_START, AUTH_ROW_STOP],
        "matched_anchor_auth_states": 9,
    }


def factor_axis_pairs(
    axis: str,
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
    pairs = []
    levels = FACTOR_SEEDS
    if axis == "recipient_A":
        for a_don in levels:
            for r_don in levels:
                for i, x in enumerate(levels):
                    for y in levels[i + 1:]:
                        pairs.append(((x, a_don, r_don), (y, a_don, r_don)))
    elif axis == "donor_A":
        for a_rec in levels:
            for r_don in levels:
                for i, x in enumerate(levels):
                    for y in levels[i + 1:]:
                        pairs.append(((a_rec, x, r_don), (a_rec, y, r_don)))
    elif axis == "donor_R":
        for a_rec in levels:
            for a_don in levels:
                for i, x in enumerate(levels):
                    for y in levels[i + 1:]:
                        pairs.append(((a_rec, a_don, x), (a_rec, a_don, y)))
    else:
        raise FactorSwapError(f"FACTOR_SWAP_AXIS:{axis}")
    require(len(pairs) == 27, f"FACTOR_SWAP_AXIS_PAIR_COUNT:{axis}:{len(pairs)}")
    return tuple(pairs)


def authenticate_repo(
    expected_head: str,
    *,
    allow_implementation_worktree: bool,
) -> None:
    branch = tb.git("branch", "--show-current")
    head = tb.git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"FACTOR_SWAP_BRANCH:{branch}")
    require(head == expected_head, f"FACTOR_SWAP_HEAD:{head}")
    require(
        tb.git_rc("merge-base", "--is-ancestor", AUTHORITY_COMMIT, expected_head)
        == 0,
        "FACTOR_SWAP_AUTHORITY_NOT_ANCESTOR",
    )
    require(
        tb.git_rc("merge-base", "--is-ancestor", SOURCE_HEAD, expected_head) == 0,
        "FACTOR_SWAP_SOURCE_NOT_ANCESTOR",
    )
    frozen_blob = tb.git(
        "rev-parse",
        f"{AUTHORITY_COMMIT}:{AUTHORITY_PATH}",
    )
    live_blob = tb.git(
        "rev-parse",
        f"HEAD:{AUTHORITY_PATH}",
    )
    require(frozen_blob == live_blob, "FACTOR_SWAP_AUTHORITY_BLOB_DRIFT")

    observed = status_paths()
    if allow_implementation_worktree:
        require(
            observed <= IMPLEMENTATION_PATHS,
            f"FACTOR_SWAP_IMPLEMENTATION_SCOPE:{sorted(observed)}",
        )
    else:
        require(
            not observed,
            f"FACTOR_SWAP_WORKTREE_NOT_CLEAN:{sorted(observed)}",
        )


def _load_shared_rows() -> torch.Tensor:
    path = ROOT / SHARED_VULNERABILITY_PATH
    require(path.is_file(), "FACTOR_SWAP_SHARED_ROWS_MISSING")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["shared_row_set"]["row_indices"]
    require(len(rows) == 120, "FACTOR_SWAP_SHARED_ROW_COUNT")
    return torch.tensor(rows, dtype=torch.long)


def load_frozen_sources() -> dict[str, Any]:
    trajectory_path = ROOT / PHASE_A_TRAJECTORY_PATH
    behavior_path = ROOT / BEHAVIORAL_COORDINATES_PATH
    synthesis_path = ROOT / SYNTHESIS_PATH

    require(trajectory_path.is_file(), "FACTOR_SWAP_TRAJECTORY_MISSING")
    require(
        sha256_file(trajectory_path) == PHASE_A_TRAJECTORY_SHA256,
        "FACTOR_SWAP_TRAJECTORY_SHA",
    )
    require(behavior_path.is_file(), "FACTOR_SWAP_BEHAVIOR_COORDS_MISSING")
    require(
        sha256_file(behavior_path) == BEHAVIORAL_COORDINATES_SHA256,
        "FACTOR_SWAP_BEHAVIOR_COORDS_SHA",
    )
    require(synthesis_path.is_file(), "FACTOR_SWAP_SYNTHESIS_MISSING")
    require(
        sha256_file(synthesis_path) == SYNTHESIS_SHA256,
        "FACTOR_SWAP_SYNTHESIS_SHA",
    )

    _phase_summary, trajectory = tb._phase_b_load_phase_a_artifacts()
    coords = torch.load(
        behavior_path,
        map_location="cpu",
        weights_only=True,
    )
    require(
        coords.get("schema_version")
        == "GEN5_AINIT_TEMPORAL_MECHANISM_BEHAVIORAL_COORDINATES_RECOVERED_V1",
        "FACTOR_SWAP_BEHAVIOR_COORDS_SCHEMA",
    )

    expected_cells = [tb.cell_name(*cell) for cell in tb.FULL_FACTORIAL_CELLS]
    require(
        list(coords["cell_order"]) == expected_cells,
        "FACTOR_SWAP_CELL_ORDER",
    )
    require(
        tuple(coords["t1_micro_state_order"]) == tb.TEMPORAL_MECHANISM_T1_STATES,
        "FACTOR_SWAP_MICRO_STATE_ORDER",
    )
    micro = coords["t1_micro_logits"]
    require(
        tuple(micro.shape) == (4, 9, DEV_ROWS, 3),
        f"FACTOR_SWAP_MICRO_SHAPE:{tuple(micro.shape)}",
    )
    b_index = tb.TEMPORAL_MECHANISM_T1_STATES.index("B_UPDATE_ONLY")
    full_index = tb.TEMPORAL_MECHANISM_T1_STATES.index("FULL_T1")
    anchor_logits = micro[b_index].contiguous()
    full_t1_logits = micro[full_index].contiguous()
    mismatch = int(
        torch.count_nonzero(
            torch.argmax(anchor_logits, dim=-1)
            != torch.argmax(full_t1_logits, dim=-1)
        ).item()
    )
    max_abs = float(torch.max(torch.abs(anchor_logits - full_t1_logits)).item())
    require(mismatch == 0, f"FACTOR_SWAP_FROZEN_BUPDATE_PRED:{mismatch}")
    require(
        max_abs <= ANCHOR_AUTH_ATOL,
        f"FACTOR_SWAP_FROZEN_BUPDATE_LOGIT:{max_abs}",
    )

    a0: dict[int, torch.Tensor] = {}
    b1: dict[tuple[int, int], torch.Tensor] = {}
    for a_seed in FACTOR_SEEDS:
        reference_a = None
        for r_seed in FACTOR_SEEDS:
            cell = (a_seed, r_seed)
            a_t0, b_t0 = tb._phase_b_snapshot(trajectory, cell, 0)
            _a_t1, b_t1 = tb._phase_b_snapshot(trajectory, cell, 1)
            require(
                int(torch.count_nonzero(b_t0).item()) == 0,
                f"FACTOR_SWAP_B0_NONZERO:{cell}",
            )
            if reference_a is None:
                reference_a = a_t0
            else:
                require(
                    torch.equal(reference_a, a_t0),
                    f"FACTOR_SWAP_A0_R_DRIFT:{a_seed}:{r_seed}",
                )
            b1[cell] = b_t1
        require(reference_a is not None, f"FACTOR_SWAP_A0_MISSING:{a_seed}")
        a0[a_seed] = reference_a

    vulnerable_rows = _load_shared_rows()
    require(
        torch.equal(vulnerable_rows, coords["vulnerable_rows"].to(torch.long)),
        "FACTOR_SWAP_VULNERABLE_ROW_IDENTITY",
    )
    return {
        "trajectory": trajectory,
        "coords": coords,
        "anchor_logits": anchor_logits,
        "full_t1_logits": full_t1_logits,
        "a0": a0,
        "b1": b1,
        "vulnerable_rows": vulnerable_rows,
        "frozen_bupdate_full_t1_mismatch": mismatch,
        "frozen_bupdate_full_t1_max_abs": max_abs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--cuda-preflight-only", action="store_true")
    modes.add_argument("--run-factor-swap", action="store_true")
    modes.add_argument("--factor-swap-worker", action="store_true")
    modes.add_argument("--merge-only", action="store_true")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker-id", type=int)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(args.implementation_freeze_commit is None, "FACTOR_SWAP_STATIC_FREEZE_ARG")
        require(args.model_snapshot is None, "FACTOR_SWAP_STATIC_MODEL_ARG")
        require(args.tokenizer_snapshot is None, "FACTOR_SWAP_STATIC_TOKENIZER_ARG")
        require(args.checkpoint is None, "FACTOR_SWAP_STATIC_CHECKPOINT_ARG")
        require(args.output_root is None, "FACTOR_SWAP_STATIC_OUTPUT_ARG")
        require(args.worker_id is None, "FACTOR_SWAP_STATIC_WORKER_ARG")
        return

    require(
        args.implementation_freeze_commit is not None,
        "FACTOR_SWAP_IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(
        args.implementation_freeze_commit == args.expected_head,
        "FACTOR_SWAP_FREEZE_HEAD_MISMATCH",
    )

    if args.merge_only:
        require(args.output_root is not None, "FACTOR_SWAP_MERGE_OUTPUT_REQUIRED")
        require(args.worker_id is None, "FACTOR_SWAP_MERGE_WORKER_FORBIDDEN")
        require(args.model_snapshot is None, "FACTOR_SWAP_MERGE_MODEL_FORBIDDEN")
        require(args.tokenizer_snapshot is None, "FACTOR_SWAP_MERGE_TOKENIZER_FORBIDDEN")
        require(args.checkpoint is None, "FACTOR_SWAP_MERGE_CHECKPOINT_FORBIDDEN")
        return

    require(args.model_snapshot is not None, "FACTOR_SWAP_MODEL_REQUIRED")
    require(args.tokenizer_snapshot is not None, "FACTOR_SWAP_TOKENIZER_REQUIRED")
    require(args.checkpoint is not None, "FACTOR_SWAP_CHECKPOINT_REQUIRED")

    if args.cuda_preflight_only:
        require(args.output_root is None, "FACTOR_SWAP_PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.worker_id is None, "FACTOR_SWAP_PREFLIGHT_WORKER_FORBIDDEN")
    elif args.run_factor_swap:
        require(args.output_root is not None, "FACTOR_SWAP_RUN_OUTPUT_REQUIRED")
        require(args.worker_id is None, "FACTOR_SWAP_RUN_WORKER_FORBIDDEN")
    elif args.factor_swap_worker:
        require(args.output_root is not None, "FACTOR_SWAP_WORKER_OUTPUT_REQUIRED")
        require(args.worker_id in (0, 1), "FACTOR_SWAP_WORKER_ID_REQUIRED")
    else:
        raise FactorSwapError("FACTOR_SWAP_MODE")


def run_static_verify(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=True,
    )
    frozen = load_frozen_sources()
    plan = planned_counts()
    require(plan["batches_per_worker"] == {"0": 14, "1": 14}, "FACTOR_SWAP_BATCH_PLAN")
    require(
        plan["cross_forward_calls_per_worker"] == {"0": 252, "1": 252},
        "FACTOR_SWAP_FORWARD_PLAN",
    )
    for axis in ("recipient_A", "donor_A", "donor_R"):
        require(len(factor_axis_pairs(axis)) == 27, f"FACTOR_SWAP_AXIS_PLAN:{axis}")
    print("GEN5_M7_FACTOR_SWAP_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print("ALL_STATES=27")
    print("NEW_CROSS_STATES=18")
    print("FULL_MATCHED_ANCHOR_RECOMPUTE=0")
    print("ROW_BATCH_SIZE=32")
    print("WORKER0_ROWS=0..419")
    print("WORKER1_ROWS=420..839")
    print("COMMON_CONTEXT_BATCHES_PER_WORKER=14")
    print("CROSS_FORWARD_CALLS_PER_WORKER=252")
    print("CROSS_FORWARD_CALLS_TOTAL=504")
    print(
        "FROZEN_BUPDATE_FULL_T1_LOGIT_MAX_ABS="
        f"{frozen['frozen_bupdate_full_t1_max_abs']:.17g}"
    )
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("FILES_WRITTEN=0")


def _runtime_bundle(args: argparse.Namespace) -> tuple[
    dict[str, Any],
    torch.nn.Module,
    Any,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    frozen = load_frozen_sources()
    (
        _summary,
        _trajectory,
        encoded,
        model,
        wrapper,
        strong_mask,
        planes,
    ) = tb._phase_b_prepare_worker_runtime(args)
    device = torch.device("cuda:0")
    features, labels, active, targets = tb.p3a._feature_batch_to_device(
        encoded["dev_bundle"],
        device,
    )
    return (
        frozen,
        model,
        wrapper,
        strong_mask,
        planes,
        features,
        labels,
        active,
        targets,
    )


def _weights_to_device(
    frozen: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    a_gpu = {
        a: tensor.to(device=device, dtype=dtype)
        for a, tensor in frozen["a0"].items()
    }
    b_gpu = {
        cell: tensor.to(device=device, dtype=dtype)
        for cell, tensor in frozen["b1"].items()
    }
    return a_gpu, b_gpu


def _recipient_latents(
    mixer_input: torch.Tensor,
    a_gpu: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    return {
        a: F.linear(mixer_input, a_gpu[a], bias=None)
        for a in FACTOR_SEEDS
    }


def _hybrid_raw_write(
    latent: torch.Tensor,
    b_weight: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    raw = F.linear(latent, b_weight, bias=None)
    return raw * attention_mask.to(raw.dtype).unsqueeze(-1)


def _forward_from_latent(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    latent: torch.Tensor,
    b_weight: torch.Tensor,
) -> torch.Tensor:
    raw = _hybrid_raw_write(
        latent,
        b_weight,
        features["attention_mask"],
    )
    return tb._phase_b_resume_from_raw_write(
        model=model,
        wrapper=wrapper,
        features=features,
        context=context,
        raw_write=raw,
    )


def run_cuda_preflight(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    gpu_meta = tb._validate_two_t4s()
    (
        frozen,
        model,
        wrapper,
        strong_mask,
        planes,
        features,
        _labels,
        active,
        targets,
    ) = _runtime_bundle(args)

    start, stop = AUTH_ROW_START, AUTH_ROW_STOP
    batch_features = tb._phase_b_batch_features(features, start, stop)
    context = tb._phase_b_prepare_common_context(
        model=model,
        wrapper=wrapper,
        features=batch_features,
        stressor_active=active[start:stop],
        target_indices=targets[start:stop],
        strong_mask=strong_mask,
        planes=planes,
    )
    device = torch.device("cuda:0")
    dtype = context["mixer_input"].dtype
    a_gpu, b_gpu = _weights_to_device(frozen, device, dtype)

    with torch.inference_mode():
        latents = _recipient_latents(context["mixer_input"], a_gpu)
        anchor = _forward_from_latent(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            context=context,
            latent=latents[6201],
            b_weight=b_gpu[(6201, 6201)],
        )
        cross = _forward_from_latent(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            context=context,
            latent=latents[6202],
            b_weight=b_gpu[(6201, 6201)],
        )

    cell_index = [tb.cell_name(*c) for c in tb.FULL_FACTORIAL_CELLS].index(
        tb.cell_name(6201, 6201)
    )
    expected = frozen["anchor_logits"][cell_index, start:stop]
    anchor_cpu = anchor.detach().cpu()
    mismatch = int(
        torch.count_nonzero(
            torch.argmax(anchor_cpu, dim=-1)
            != torch.argmax(expected, dim=-1)
        ).item()
    )
    max_abs = float(torch.max(torch.abs(anchor_cpu - expected)).item())
    require(mismatch == 0, f"FACTOR_SWAP_PREFLIGHT_ANCHOR_PRED:{mismatch}")
    require(max_abs <= ANCHOR_AUTH_ATOL, f"FACTOR_SWAP_PREFLIGHT_ANCHOR_LOGIT:{max_abs}")
    require(torch.isfinite(cross).all().item(), "FACTOR_SWAP_PREFLIGHT_CROSS_NONFINITE")
    parameter_gradients = any(p.grad is not None for p in model.parameters())
    require(not parameter_gradients, "FACTOR_SWAP_PREFLIGHT_PARAMETER_GRAD")

    print("GEN5_M7_FACTOR_SWAP_CUDA_PREFLIGHT_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"GPU_COUNT={gpu_meta['gpu_count']}")
    print("ROW_BATCH_SIZE=32")
    print("COMMON_CONTEXT_FORWARD_BATCHES=1")
    print("MATCHED_ANCHOR_FORWARD_STATES=1")
    print("CROSS_A_FORWARD_STATES=1")
    print(f"ANCHOR_PREDICTION_MISMATCH={mismatch}")
    print(f"ANCHOR_LOGIT_MAX_ABS={max_abs:.17g}")
    print("CROSS_OUTPUT_FINITE=True")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("PREFLIGHT_COLLECTION=FORBIDDEN")
    print("FILES_WRITTEN=0")


def _run_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "GEN5_M7_FACTOR_SWAP_RUN_IDENTITY_V1",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "authority_commit": AUTHORITY_COMMIT,
        "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
        "behavioral_coordinates_sha256": BEHAVIORAL_COORDINATES_SHA256,
        "row_batch_size": ROW_BATCH_SIZE,
        "worker_ranges": {
            str(k): list(v) for k, v in WORKER_RANGES.items()
        },
        "cross_hybrid_order": [
            hybrid_name(*state) for state in CROSS_HYBRIDS
        ],
        "all_hybrid_order": [
            hybrid_name(*state) for state in ALL_HYBRIDS
        ],
    }


def _init_or_validate_run_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    identity_path = root / "run_identity.json"
    expected = _run_identity(args)
    if not root.exists():
        root.mkdir(parents=True)
        atomic_bytes(identity_path, canonical_json_bytes(expected))
    else:
        require(root.is_dir(), "FACTOR_SWAP_OUTPUT_NOT_DIR")
        require(identity_path.is_file(), "FACTOR_SWAP_RUN_IDENTITY_MISSING")
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        require(observed == expected, "FACTOR_SWAP_RUN_IDENTITY_MISMATCH")
    return root


def _chunk_path(output_root: Path, worker_id: int, start: int, stop: int) -> Path:
    return output_root / f"worker{worker_id}" / f"chunk_rows_{start:03d}_{stop:03d}.pt"


def _validate_chunk(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    worker_id: int,
    start: int,
    stop: int,
) -> None:
    require(
        payload.get("schema_version") == "GEN5_M7_FACTOR_SWAP_CHUNK_V1",
        "FACTOR_SWAP_CHUNK_SCHEMA",
    )
    require(payload.get("execution_head") == args.expected_head, "FACTOR_SWAP_CHUNK_HEAD")
    require(
        payload.get("implementation_freeze_commit") == args.implementation_freeze_commit,
        "FACTOR_SWAP_CHUNK_FREEZE",
    )
    require(int(payload.get("worker_id", -1)) == worker_id, "FACTOR_SWAP_CHUNK_WORKER")
    require(list(payload.get("row_interval", ())) == [start, stop], "FACTOR_SWAP_CHUNK_ROWS")
    require(
        list(payload.get("hybrid_order", ()))
        == [hybrid_name(*state) for state in CROSS_HYBRIDS],
        "FACTOR_SWAP_CHUNK_HYBRID_ORDER",
    )
    logits = payload["logits"]
    require(
        tuple(logits.shape) == (len(CROSS_HYBRIDS), stop - start, 3),
        f"FACTOR_SWAP_CHUNK_LOGIT_SHAPE:{tuple(logits.shape)}",
    )
    require(torch.isfinite(logits).all().item(), "FACTOR_SWAP_CHUNK_NONFINITE")


def _write_chunk(
    path: Path,
    *,
    args: argparse.Namespace,
    worker_id: int,
    start: int,
    stop: int,
    logits: torch.Tensor,
) -> None:
    payload = {
        "schema_version": "GEN5_M7_FACTOR_SWAP_CHUNK_V1",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "worker_id": worker_id,
        "row_interval": [start, stop],
        "hybrid_order": [hybrid_name(*state) for state in CROSS_HYBRIDS],
        "logits": logits.contiguous(),
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "parameter_gradients_accumulated": False,
        "confirmatory_9601_9900_loaded": False,
    }
    atomic_torch_save(path, payload)


def _load_valid_chunk(
    path: Path,
    *,
    args: argparse.Namespace,
    worker_id: int,
    start: int,
    stop: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_chunk(
        payload,
        args=args,
        worker_id=worker_id,
        start=start,
        stop=stop,
    )
    return payload


def _write_worker_manifest(
    output_root: Path,
    *,
    args: argparse.Namespace,
    worker_id: int,
) -> None:
    rows = []
    for start, stop in worker_batches(worker_id):
        path = _chunk_path(output_root, worker_id, start, stop)
        if path.is_file():
            payload = _load_valid_chunk(
                path,
                args=args,
                worker_id=worker_id,
                start=start,
                stop=stop,
            )
            del payload
            rows.append({
                "path": path.relative_to(output_root).as_posix(),
                "row_interval": [start, stop],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        "schema_version": "GEN5_M7_FACTOR_SWAP_WORKER_MANIFEST_V1",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "worker_id": worker_id,
        "expected_chunk_count": len(worker_batches(worker_id)),
        "completed_chunk_count": len(rows),
        "chunks": rows,
    }
    atomic_bytes(
        output_root / f"worker{worker_id}" / "chunk_manifest.json",
        canonical_json_bytes(manifest),
    )


def _anchor_auth_path(output_root: Path) -> Path:
    return output_root / "worker0" / "anchor_auth.pt"


def _validate_anchor_auth(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    require(
        payload.get("schema_version") == "GEN5_M7_FACTOR_SWAP_ANCHOR_AUTH_V1",
        "FACTOR_SWAP_ANCHOR_AUTH_SCHEMA",
    )
    require(payload.get("execution_head") == args.expected_head, "FACTOR_SWAP_ANCHOR_AUTH_HEAD")
    require(
        payload.get("implementation_freeze_commit") == args.implementation_freeze_commit,
        "FACTOR_SWAP_ANCHOR_AUTH_FREEZE",
    )
    require(
        list(payload.get("row_interval", ())) == [AUTH_ROW_START, AUTH_ROW_STOP],
        "FACTOR_SWAP_ANCHOR_AUTH_ROWS",
    )
    logits = payload["logits"]
    require(tuple(logits.shape) == (9, 32, 3), "FACTOR_SWAP_ANCHOR_AUTH_SHAPE")
    require(int(payload["prediction_mismatch_count"]) == 0, "FACTOR_SWAP_ANCHOR_AUTH_PRED")
    require(
        float(payload["logit_max_abs"]) <= ANCHOR_AUTH_ATOL,
        "FACTOR_SWAP_ANCHOR_AUTH_LOGIT",
    )


def _consolidate_worker(
    *,
    args: argparse.Namespace,
    output_root: Path,
    worker_id: int,
) -> dict[str, Any]:
    start0, stop0 = WORKER_RANGES[worker_id]
    logits = torch.empty(
        (len(CROSS_HYBRIDS), stop0 - start0, 3),
        dtype=torch.float32,
    )
    chunk_rows = []
    for start, stop in worker_batches(worker_id):
        path = _chunk_path(output_root, worker_id, start, stop)
        require(path.is_file(), f"FACTOR_SWAP_CHUNK_MISSING:{path}")
        payload = _load_valid_chunk(
            path,
            args=args,
            worker_id=worker_id,
            start=start,
            stop=stop,
        )
        logits[:, start - start0: stop - start0] = payload["logits"]
        chunk_rows.append({
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
            "row_interval": [start, stop],
        })

    result = {
        "schema_version": "GEN5_M7_FACTOR_SWAP_WORKER_V1",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "worker_id": worker_id,
        "row_interval": [start0, stop0],
        "hybrid_order": [hybrid_name(*state) for state in CROSS_HYBRIDS],
        "logits": logits,
        "chunks": chunk_rows,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "parameter_gradients_accumulated": False,
        "confirmatory_9601_9900_loaded": False,
    }
    path = output_root / f"worker{worker_id}" / "worker_result.pt"
    atomic_torch_save(path, result)
    return result


def run_worker(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    require(torch.cuda.is_available(), "FACTOR_SWAP_WORKER_CUDA")
    require(torch.cuda.device_count() == 1, "FACTOR_SWAP_WORKER_VISIBLE_GPU_COUNT")
    worker_id = int(args.worker_id)
    output_root = _init_or_validate_run_root(args)
    worker_root = output_root / f"worker{worker_id}"
    worker_root.mkdir(parents=True, exist_ok=True)

    missing_batches = []
    for start, stop in worker_batches(worker_id):
        path = _chunk_path(output_root, worker_id, start, stop)
        if path.is_file():
            _load_valid_chunk(
                path,
                args=args,
                worker_id=worker_id,
                start=start,
                stop=stop,
            )
        else:
            missing_batches.append((start, stop))

    anchor_path = _anchor_auth_path(output_root)
    anchor_needed = worker_id == 0 and not anchor_path.is_file()
    if worker_id == 0 and anchor_path.is_file():
        _validate_anchor_auth(
            torch.load(anchor_path, map_location="cpu", weights_only=True),
            args=args,
        )

    if missing_batches or anchor_needed:
        (
            frozen,
            model,
            wrapper,
            strong_mask,
            planes,
            features,
            _labels,
            active,
            targets,
        ) = _runtime_bundle(args)
        device = torch.device("cuda:0")
        a_gpu = None
        b_gpu = None

        batches_to_run = list(missing_batches)
        if anchor_needed and (AUTH_ROW_START, AUTH_ROW_STOP) not in batches_to_run:
            batches_to_run.insert(0, (AUTH_ROW_START, AUTH_ROW_STOP))

        for start, stop in batches_to_run:
            batch_features = tb._phase_b_batch_features(features, start, stop)
            context = tb._phase_b_prepare_common_context(
                model=model,
                wrapper=wrapper,
                features=batch_features,
                stressor_active=active[start:stop],
                target_indices=targets[start:stop],
                strong_mask=strong_mask,
                planes=planes,
            )
            if a_gpu is None or b_gpu is None:
                a_gpu, b_gpu = _weights_to_device(
                    frozen,
                    device,
                    context["mixer_input"].dtype,
                )
            with torch.inference_mode():
                latents = _recipient_latents(context["mixer_input"], a_gpu)

                chunk_path = _chunk_path(output_root, worker_id, start, stop)
                if (start, stop) in missing_batches:
                    chunk_logits = torch.empty(
                        (len(CROSS_HYBRIDS), stop - start, 3),
                        dtype=torch.float32,
                    )
                    for index, (a_rec, a_don, r_don) in enumerate(CROSS_HYBRIDS):
                        live = _forward_from_latent(
                            model=model,
                            wrapper=wrapper,
                            features=batch_features,
                            context=context,
                            latent=latents[a_rec],
                            b_weight=b_gpu[(a_don, r_don)],
                        )
                        chunk_logits[index] = live.detach().cpu()
                    _write_chunk(
                        chunk_path,
                        args=args,
                        worker_id=worker_id,
                        start=start,
                        stop=stop,
                        logits=chunk_logits,
                    )
                    _write_worker_manifest(
                        output_root,
                        args=args,
                        worker_id=worker_id,
                    )

                if (
                    worker_id == 0
                    and start == AUTH_ROW_START
                    and stop == AUTH_ROW_STOP
                    and anchor_needed
                ):
                    anchor_logits = torch.empty((9, 32, 3), dtype=torch.float32)
                    cell_order = list(tb.FULL_FACTORIAL_CELLS)
                    for index, (a_seed, r_seed) in enumerate(cell_order):
                        live = _forward_from_latent(
                            model=model,
                            wrapper=wrapper,
                            features=batch_features,
                            context=context,
                            latent=latents[a_seed],
                            b_weight=b_gpu[(a_seed, r_seed)],
                        )
                        anchor_logits[index] = live.detach().cpu()
                    expected = frozen["anchor_logits"][:, AUTH_ROW_START:AUTH_ROW_STOP]
                    mismatch = int(
                        torch.count_nonzero(
                            torch.argmax(anchor_logits, dim=-1)
                            != torch.argmax(expected, dim=-1)
                        ).item()
                    )
                    max_abs = float(
                        torch.max(torch.abs(anchor_logits - expected)).item()
                    )
                    require(mismatch == 0, f"FACTOR_SWAP_ANCHOR_AUTH_PRED:{mismatch}")
                    require(
                        max_abs <= ANCHOR_AUTH_ATOL,
                        f"FACTOR_SWAP_ANCHOR_AUTH_LOGIT:{max_abs}",
                    )
                    auth_payload = {
                        "schema_version": "GEN5_M7_FACTOR_SWAP_ANCHOR_AUTH_V1",
                        "execution_head": args.expected_head,
                        "implementation_freeze_commit": args.implementation_freeze_commit,
                        "row_interval": [AUTH_ROW_START, AUTH_ROW_STOP],
                        "cell_order": [tb.cell_name(*cell) for cell in cell_order],
                        "logits": anchor_logits,
                        "prediction_mismatch_count": mismatch,
                        "logit_max_abs": max_abs,
                    }
                    atomic_torch_save(anchor_path, auth_payload)

        parameter_gradients = any(p.grad is not None for p in model.parameters())
        require(not parameter_gradients, "FACTOR_SWAP_WORKER_PARAMETER_GRAD")

    _write_worker_manifest(
        output_root,
        args=args,
        worker_id=worker_id,
    )
    result = _consolidate_worker(
        args=args,
        output_root=output_root,
        worker_id=worker_id,
    )
    if worker_id == 0:
        require(anchor_path.is_file(), "FACTOR_SWAP_ANCHOR_AUTH_MISSING")
        _validate_anchor_auth(
            torch.load(anchor_path, map_location="cpu", weights_only=True),
            args=args,
        )

    print(
        "GEN5_M7_FACTOR_SWAP_WORKER_PASS "
        f"worker={worker_id} "
        f"rows={WORKER_RANGES[worker_id][0]}..{WORKER_RANGES[worker_id][1]-1} "
        f"chunks={len(worker_batches(worker_id))} "
        f"cross_states={len(CROSS_HYBRIDS)}"
    )
    print(f"WORKER_RESULT_SHA256={sha256_file(output_root / f'worker{worker_id}' / 'worker_result.pt')}")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_EXECUTED=False")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")


def _spawn_workers(args: argparse.Namespace, output_root: Path) -> None:
    processes = []
    script = Path(__file__).resolve()
    for worker_id in (0, 1):
        worker_root = output_root / f"worker{worker_id}"
        worker_root.mkdir(parents=True, exist_ok=True)
        log_path = worker_root / "worker.log"
        command = [
            sys.executable,
            str(script),
            "--factor-swap-worker",
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
            "--output-root",
            str(output_root),
            "--worker-id",
            str(worker_id),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_id, process, handle, log_path))

    codes = {}
    for worker_id, process, handle, _log in processes:
        codes[worker_id] = process.wait()
        handle.close()
    if any(code != 0 for code in codes.values()):
        for worker_id, _process, _handle, log_path in processes:
            print(f"=== FACTOR SWAP WORKER {worker_id} LOG ===")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
        raise FactorSwapError(f"FACTOR_SWAP_WORKER_FAILURE:{codes}")


def _coordinate_affinity(
    hybrid: torch.Tensor,
    recipient: torch.Tensor,
    donor: torch.Tensor,
) -> dict[str, Any]:
    d_rec = torch.linalg.vector_norm(hybrid - recipient, dim=-1).to(torch.float64)
    d_don = torch.linalg.vector_norm(hybrid - donor, dim=-1).to(torch.float64)
    affinity = (d_don - d_rec) / (d_don + d_rec + AFFINITY_EPS)
    return {
        "d_rec_mean": float(d_rec.mean().item()),
        "d_rec_median": float(d_rec.median().item()),
        "d_don_mean": float(d_don.mean().item()),
        "d_don_median": float(d_don.median().item()),
        "affinity_mean": float(affinity.mean().item()),
        "affinity_median": float(affinity.median().item()),
        "affinity_positive_count": int(torch.count_nonzero(affinity > 0).item()),
        "affinity_negative_count": int(torch.count_nonzero(affinity < 0).item()),
        "affinity_zero_count": int(torch.count_nonzero(affinity == 0).item()),
    }


def _segment_diagnostic(
    hybrid: torch.Tensor,
    recipient: torch.Tensor,
    donor: torch.Tensor,
) -> dict[str, Any]:
    direction = donor - recipient
    numerator = torch.sum((hybrid - recipient) * direction, dim=-1).to(torch.float64)
    denom = torch.sum(direction * direction, dim=-1).to(torch.float64)
    alpha = torch.where(
        denom > 0,
        numerator / denom,
        torch.zeros_like(numerator),
    )
    return {
        "alpha_mean": float(alpha.mean().item()),
        "alpha_median": float(alpha.median().item()),
        "outside_segment_count": int(
            torch.count_nonzero((alpha < 0) | (alpha > 1)).item()
        ),
        "degenerate_reference_count": int(torch.count_nonzero(denom == 0).item()),
    }


def _subset(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return value.index_select(0, rows)


def _pair_metric_row(
    *,
    state: tuple[int, int, int],
    logits_by_state: Mapping[tuple[int, int, int], torch.Tensor],
    vulnerable_rows: torch.Tensor,
) -> dict[str, Any]:
    a_rec, a_don, r_don = state
    x_logits = logits_by_state[state]
    r_logits = logits_by_state[(a_rec, a_rec, r_don)]
    d_logits = logits_by_state[(a_don, a_don, r_don)]

    x_center = tb.temporal_mechanism_centered_logits(x_logits)
    r_center = tb.temporal_mechanism_centered_logits(r_logits)
    d_center = tb.temporal_mechanism_centered_logits(d_logits)
    x_margin = tb.temporal_mechanism_margin_vector(x_logits)
    r_margin = tb.temporal_mechanism_margin_vector(r_logits)
    d_margin = tb.temporal_mechanism_margin_vector(d_logits)

    x_pred = torch.argmax(x_logits, dim=-1)
    r_pred = torch.argmax(r_logits, dim=-1)
    d_pred = torch.argmax(d_logits, dim=-1)

    def pred_diag(rows: torch.Tensor | None) -> dict[str, int]:
        xp = x_pred if rows is None else _subset(x_pred, rows)
        rp = r_pred if rows is None else _subset(r_pred, rows)
        dp = d_pred if rows is None else _subset(d_pred, rows)
        return {
            "recipient_agreement": int(torch.count_nonzero(xp == rp).item()),
            "donor_agreement": int(torch.count_nonzero(xp == dp).item()),
            "agree_neither": int(
                torch.count_nonzero((xp != rp) & (xp != dp)).item()
            ),
            "row_count": int(xp.numel()),
        }

    result = {
        "hybrid": hybrid_name(*state),
        "a_rec": a_rec,
        "a_don": a_don,
        "r_don": r_don,
        "recipient_reference": hybrid_name(a_rec, a_rec, r_don),
        "donor_reference": hybrid_name(a_don, a_don, r_don),
        "all_840": {
            "centered_logits": _coordinate_affinity(x_center, r_center, d_center),
            "two_margins": _coordinate_affinity(x_margin, r_margin, d_margin),
            "prediction": pred_diag(None),
            "segment_two_margins": _segment_diagnostic(
                x_margin, r_margin, d_margin
            ),
            "hybrid_margin_norm_min": float(
                torch.linalg.vector_norm(x_margin, dim=-1).min().item()
            ),
            "hybrid_margin_norm_max": float(
                torch.linalg.vector_norm(x_margin, dim=-1).max().item()
            ),
        },
        "vulnerable_120": {
            "centered_logits": _coordinate_affinity(
                _subset(x_center, vulnerable_rows),
                _subset(r_center, vulnerable_rows),
                _subset(d_center, vulnerable_rows),
            ),
            "two_margins": _coordinate_affinity(
                _subset(x_margin, vulnerable_rows),
                _subset(r_margin, vulnerable_rows),
                _subset(d_margin, vulnerable_rows),
            ),
            "prediction": pred_diag(vulnerable_rows),
            "segment_two_margins": _segment_diagnostic(
                _subset(x_margin, vulnerable_rows),
                _subset(r_margin, vulnerable_rows),
                _subset(d_margin, vulnerable_rows),
            ),
        },
    }
    return result


def _factor_effect_summary(
    *,
    logits_by_state: Mapping[tuple[int, int, int], torch.Tensor],
) -> dict[str, Any]:
    margins = {
        state: tb.temporal_mechanism_margin_vector(logits)
        for state, logits in logits_by_state.items()
    }
    predictions = {
        state: torch.argmax(logits, dim=-1)
        for state, logits in logits_by_state.items()
    }
    result = {}
    for axis in ("recipient_A", "donor_A", "donor_R"):
        rows = []
        for left, right in factor_axis_pairs(axis):
            per_row = torch.linalg.vector_norm(
                margins[right] - margins[left],
                dim=-1,
            ).to(torch.float64)
            rows.append({
                "left": hybrid_name(*left),
                "right": hybrid_name(*right),
                "mean_margin_l2": float(per_row.mean().item()),
                "prediction_disagreement_count": int(
                    torch.count_nonzero(
                        predictions[left] != predictions[right]
                    ).item()
                ),
            })
        mean_values = [row["mean_margin_l2"] for row in rows]
        disagreements = [row["prediction_disagreement_count"] for row in rows]
        result[axis] = {
            "pair_count": len(rows),
            "mean_pair_margin_l2": sum(mean_values) / len(mean_values),
            "pair_margin_l2_min": min(mean_values),
            "pair_margin_l2_max": max(mean_values),
            "prediction_disagreement_mean": sum(disagreements) / len(disagreements),
            "prediction_disagreement_min": min(disagreements),
            "prediction_disagreement_max": max(disagreements),
            "pairs": rows,
        }
    return result


def _load_worker_result(
    *,
    args: argparse.Namespace,
    output_root: Path,
    worker_id: int,
) -> dict[str, Any]:
    path = output_root / f"worker{worker_id}" / "worker_result.pt"
    if not path.is_file():
        return _consolidate_worker(
            args=args,
            output_root=output_root,
            worker_id=worker_id,
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        payload.get("schema_version") == "GEN5_M7_FACTOR_SWAP_WORKER_V1",
        "FACTOR_SWAP_WORKER_SCHEMA",
    )
    require(payload.get("execution_head") == args.expected_head, "FACTOR_SWAP_WORKER_HEAD")
    require(int(payload["worker_id"]) == worker_id, "FACTOR_SWAP_WORKER_ID")
    require(
        list(payload["hybrid_order"])
        == [hybrid_name(*state) for state in CROSS_HYBRIDS],
        "FACTOR_SWAP_WORKER_HYBRID_ORDER",
    )
    start, stop = WORKER_RANGES[worker_id]
    require(
        list(payload["row_interval"]) == [start, stop],
        "FACTOR_SWAP_WORKER_ROWS",
    )
    require(
        tuple(payload["logits"].shape) == (18, stop - start, 3),
        "FACTOR_SWAP_WORKER_LOGIT_SHAPE",
    )
    for key in (
        "training_executed",
        "backward_executed",
        "optimizer_constructed",
        "optimizer_step_executed",
        "parameter_gradients_accumulated",
        "confirmatory_9601_9900_loaded",
    ):
        require(payload.get(key) is False, f"FACTOR_SWAP_WORKER_FLAG:{key}")
    return payload


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = b"".join(canonical_json_bytes(row) for row in rows)
    atomic_bytes(path, data)


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_root).as_posix()
        if rel == "artifact_manifest.json":
            continue
        files[rel] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "GEN5_M7_FACTOR_SWAP_ARTIFACT_MANIFEST_V1",
        "files": files,
    }


def merge_only(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    output_root = _init_or_validate_run_root(args)
    frozen = load_frozen_sources()

    workers = [
        _load_worker_result(args=args, output_root=output_root, worker_id=0),
        _load_worker_result(args=args, output_root=output_root, worker_id=1),
    ]
    cross_logits = torch.empty((18, DEV_ROWS, 3), dtype=torch.float32)
    for worker in workers:
        start, stop = worker["row_interval"]
        cross_logits[:, start:stop] = worker["logits"]

    anchor_auth_path = _anchor_auth_path(output_root)
    require(anchor_auth_path.is_file(), "FACTOR_SWAP_ANCHOR_AUTH_MISSING")
    anchor_auth = torch.load(
        anchor_auth_path,
        map_location="cpu",
        weights_only=True,
    )
    _validate_anchor_auth(anchor_auth, args=args)

    cell_order = list(tb.FULL_FACTORIAL_CELLS)
    cell_index = {cell: i for i, cell in enumerate(cell_order)}
    cross_index = {state: i for i, state in enumerate(CROSS_HYBRIDS)}

    logits_by_state: dict[tuple[int, int, int], torch.Tensor] = {}
    provenance_kind = []
    state_logits = []
    for state in ALL_HYBRIDS:
        a_rec, a_don, r_don = state
        if a_rec == a_don:
            logits = frozen["anchor_logits"][cell_index[(a_rec, r_don)]]
            provenance_kind.append("FROZEN_MATCHED_REFERENCE")
        else:
            logits = cross_logits[cross_index[state]]
            provenance_kind.append("NEW_CROSS_A_INTERVENTION")
        logits_by_state[state] = logits
        state_logits.append(logits)

    all_logits = torch.stack(state_logits, dim=0)
    centered = tb.temporal_mechanism_centered_logits(
        all_logits.reshape(-1, 3)
    ).reshape(27, DEV_ROWS, 3)
    margins = tb.temporal_mechanism_margin_vector(
        all_logits.reshape(-1, 3)
    ).reshape(27, DEV_ROWS, 2)
    predictions = torch.argmax(all_logits, dim=-1)

    pair_rows = [
        _pair_metric_row(
            state=state,
            logits_by_state=logits_by_state,
            vulnerable_rows=frozen["vulnerable_rows"],
        )
        for state in CROSS_HYBRIDS
    ]
    factor_effects = _factor_effect_summary(logits_by_state=logits_by_state)

    logits_path = output_root / "factor_swap_logits.pt"
    atomic_torch_save(
        logits_path,
        {
            "schema_version": "GEN5_M7_FACTOR_SWAP_LOGITS_V1",
            "execution_head": args.expected_head,
            "implementation_freeze_commit": args.implementation_freeze_commit,
            "state_order": [hybrid_name(*state) for state in ALL_HYBRIDS],
            "provenance_kind": provenance_kind,
            "logits": all_logits,
            "centered_logits": centered,
            "two_margins": margins,
            "predictions": predictions,
            "vulnerable_rows": frozen["vulnerable_rows"],
        },
    )

    pair_path = output_root / "factor_swap_pair_metrics.jsonl"
    _write_jsonl_atomic(pair_path, pair_rows)

    recipient_mean = factor_effects["recipient_A"]["mean_pair_margin_l2"]
    donor_mean = factor_effects["donor_A"]["mean_pair_margin_l2"]
    rng_mean = factor_effects["donor_R"]["mean_pair_margin_l2"]

    summary = {
        "schema_version": "GEN5_M7_FACTOR_SWAP_SUMMARY_V1",
        "result": "PASS_GEN5_M7_FACTOR_SWAP",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "authority_commit": AUTHORITY_COMMIT,
        "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
        "behavioral_coordinates_sha256": BEHAVIORAL_COORDINATES_SHA256,
        "dev_rows": DEV_ROWS,
        "vulnerable_rows": 120,
        "all_state_count": 27,
        "frozen_matched_reference_count": 9,
        "new_cross_intervention_count": 18,
        "row_batch_size": ROW_BATCH_SIZE,
        "worker_partition": {
            "0": [0, 420],
            "1": [420, 840],
        },
        "planned_counts": planned_counts(),
        "frozen_bupdate_full_t1_prediction_mismatch": (
            frozen["frozen_bupdate_full_t1_mismatch"]
        ),
        "frozen_bupdate_full_t1_logit_max_abs": (
            frozen["frozen_bupdate_full_t1_max_abs"]
        ),
        "matched_anchor_auth": {
            "row_interval": [AUTH_ROW_START, AUTH_ROW_STOP],
            "prediction_mismatch_count": int(
                anchor_auth["prediction_mismatch_count"]
            ),
            "logit_max_abs": float(anchor_auth["logit_max_abs"]),
        },
        "factor_effects": factor_effects,
        "aggregate_factor_contrast": {
            "recipient_A_mean_margin_l2": recipient_mean,
            "donor_A_mean_margin_l2": donor_mean,
            "donor_R_mean_margin_l2": rng_mean,
            "recipient_over_donor_A": (
                None if donor_mean == 0 else recipient_mean / donor_mean
            ),
            "donor_A_over_donor_R": (
                None if rng_mean == 0 else donor_mean / rng_mean
            ),
            "recipient_A_over_donor_R": (
                None if rng_mean == 0 else recipient_mean / rng_mean
            ),
        },
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "parameter_gradients_accumulated": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
    }
    summary_path = output_root / "factor_swap_summary.json"
    atomic_bytes(summary_path, canonical_json_bytes(summary))

    provenance = {
        "schema_version": "GEN5_M7_FACTOR_SWAP_PROVENANCE_V1",
        "status": "PASS",
        "authority_commit": AUTHORITY_COMMIT,
        "authority_blob": tb.git("rev-parse", f"HEAD:{AUTHORITY_PATH}"),
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
        "behavioral_coordinates_sha256": BEHAVIORAL_COORDINATES_SHA256,
        "factor_swap_logits_sha256": sha256_file(logits_path),
        "factor_swap_pair_metrics_sha256": sha256_file(pair_path),
        "factor_swap_summary_sha256": sha256_file(summary_path),
        "worker0_result_sha256": sha256_file(
            output_root / "worker0" / "worker_result.pt"
        ),
        "worker1_result_sha256": sha256_file(
            output_root / "worker1" / "worker_result.pt"
        ),
        "anchor_auth_sha256": sha256_file(anchor_auth_path),
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "parameter_gradients_accumulated": False,
        "confirmatory_9601_9900_loaded": False,
        "merge_only_recoverable": True,
    }
    provenance_path = output_root / "run_provenance.json"
    atomic_bytes(provenance_path, canonical_json_bytes(provenance))

    manifest_path = output_root / "artifact_manifest.json"
    atomic_bytes(
        manifest_path,
        canonical_json_bytes(_artifact_manifest(output_root)),
    )

    print("GEN5_M7_FACTOR_SWAP_MERGE_PASS")
    print(f"HEAD={args.expected_head}")
    print("ALL_STATES=27")
    print("FROZEN_MATCHED_REFERENCES=9")
    print("NEW_CROSS_A_INTERVENTIONS=18")
    print("MATCHED_ANCHOR_AUTH_PREDICTION_MISMATCH=0")
    print(
        "MATCHED_ANCHOR_AUTH_LOGIT_MAX_ABS="
        f"{float(anchor_auth['logit_max_abs']):.17g}"
    )
    print(f"RECIPIENT_A_MEAN_MARGIN_L2={recipient_mean:.17g}")
    print(f"DONOR_A_MEAN_MARGIN_L2={donor_mean:.17g}")
    print(f"DONOR_R_MEAN_MARGIN_L2={rng_mean:.17g}")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_EXECUTED=False")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={summary_path}")
    print(f"LOGITS={logits_path}")
    print(f"PAIR_METRICS={pair_path}")
    print(f"PROVENANCE={provenance_path}")
    print(f"MANIFEST={manifest_path}")


def run_factor_swap(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    gpu_meta = tb._validate_two_t4s()
    output_root = _init_or_validate_run_root(args)
    plan = planned_counts()
    print("GEN5_M7_FACTOR_SWAP_EXECUTION_PLAN")
    print(f"ROW_BATCH_SIZE={ROW_BATCH_SIZE}")
    print(f"COMMON_CONTEXT_BATCHES_PER_WORKER={plan['batches_per_worker']['0']}")
    print(f"CROSS_FORWARD_CALLS_PER_WORKER={plan['cross_forward_calls_per_worker']['0']}")
    print(f"CROSS_FORWARD_CALLS_TOTAL={plan['cross_forward_calls_total']}")
    print("FULL_MATCHED_ANCHOR_RECOMPUTE=0")
    print(f"GPU_COUNT={gpu_meta['gpu_count']}")
    _spawn_workers(args, output_root)
    merge_only(args)
    print("GEN5_M7_FACTOR_SWAP_RUN_PASS")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.static_verify_only:
        run_static_verify(args)
    elif args.cuda_preflight_only:
        run_cuda_preflight(args)
    elif args.run_factor_swap:
        run_factor_swap(args)
    elif args.factor_swap_worker:
        run_worker(args)
    elif args.merge_only:
        merge_only(args)
    else:
        raise FactorSwapError("FACTOR_SWAP_NO_MODE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
