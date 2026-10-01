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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

STAGE_B_FREEZE_COMMIT = "b49c1339231a1c5dbb7f870ddc81159109b08c1e"
PHASE3A_EXECUTION_COMMIT = "d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e"

STAGE_B_SUMMARY = Path(
    "reports/reason_router_gen5_optimization_path_bypass_stage_b_runs/"
    "gen5-stageb-functional-decomposition-f70c0b2-r2/"
    "functional_decomposition_summary.json"
)
STAGE_B_PROVENANCE = Path(
    "reports/reason_router_gen5_optimization_path_bypass_stage_b_runs/"
    "gen5-stageb-functional-decomposition-f70c0b2-r2/"
    "run_provenance.json"
)

EXECUTION_AUTHORITY_PATH = Path(
    "reports/reason_router_gen5_optimization_path_bypass_"
    "stage_c_step0_gradient_execution_authority_spec_candidate.md"
)

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/reason_router_gen5_optimization_path_bypass_stage_c_step0_gradient_geometry.py",
    "tests/test_reason_router_gen5_optimization_path_bypass_stage_c_step0_gradient_geometry.py",
})

SEEDS = (6201, 6202, 6203)
PRESSURES = ("P0", "PR", "PC")
CELLS = tuple((seed, pressure) for seed in SEEDS for pressure in PRESSURES)

GPU_QUEUES = {
    0: (
        (6201, "P0"),
        (6201, "PC"),
        (6202, "PR"),
        (6203, "P0"),
        (6203, "PC"),
    ),
    1: (
        (6201, "PR"),
        (6202, "P0"),
        (6202, "PC"),
        (6203, "PR"),
    ),
}

TRAIN_ROWS = 3360
STREAM_ROWS = 240
STEP0_LOSS_ATOL = 1e-6
FLOAT32_EPS = float(torch.finfo(torch.float32).eps)
AMBIENT_DIM = 24576
RANK2_RANDOM_AFFINITY = 2.0 / AMBIENT_DIM

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)


class StageCError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StageCError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StageCError("GIT_FAILURE:" + " ".join(args)) from exc


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
    )
    out: set[str] = set()
    for line in raw.splitlines():
        if not line:
            continue
        require(len(line) >= 4, f"STATUS_ROW:{line}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out.add(path)
    return out


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
        git_rc(
            "merge-base",
            "--is-ancestor",
            STAGE_B_FREEZE_COMMIT,
            expected_head,
        ) == 0,
        "STAGE_B_FREEZE_NOT_ANCESTOR",
    )

    observed = status_paths()
    if allow_implementation_worktree:
        require(
            observed <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(observed)}",
        )
    else:
        require(not observed, f"WORKTREE_NOT_CLEAN:{sorted(observed)}")


def validate_execution_authority(
    *,
    expected_head: str,
    implementation_freeze_commit: str,
    execution_authority_commit: str,
) -> None:
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            implementation_freeze_commit,
            execution_authority_commit,
        ) == 0,
        "IMPLEMENTATION_FREEZE_NOT_ANCESTOR",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            execution_authority_commit,
            expected_head,
        ) == 0,
        "EXECUTION_AUTHORITY_NOT_ANCESTOR",
    )

    for rel in sorted(AUTHORIZED_IMPLEMENTATION_PATHS):
        require(
            git_rc(
                "diff",
                "--quiet",
                implementation_freeze_commit,
                expected_head,
                "--",
                rel,
            ) == 0,
            f"IMPLEMENTATION_DRIFT:{rel}",
        )

    rel = str(EXECUTION_AUTHORITY_PATH).replace("\\", "/")
    require((ROOT / EXECUTION_AUTHORITY_PATH).is_file(), "AUTHORITY_MISSING")
    require(
        git("rev-parse", f"{execution_authority_commit}:{rel}")
        == git("rev-parse", f"HEAD:{rel}"),
        "AUTHORITY_BLOB_DRIFT",
    )
    text = git("show", f"{execution_authority_commit}:{rel}")
    tokens = (
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_BACKWARD_ONLY_STAGE_C_STEP0_GRADIENT_MATRIX",
        f"IMPLEMENTATION_FREEZE_COMMIT={implementation_freeze_commit}",
        "TRAINING_ALLOWED=NO",
        "OPTIMIZER_ALLOWED=NO",
        "OPTIMIZER_STEP_ALLOWED=NO",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
    )
    for token in tokens:
        require(token in text, f"AUTHORITY_TOKEN:{token}")


def validate_stage_b_freeze() -> dict[str, Any]:
    summary_path = ROOT / STAGE_B_SUMMARY
    provenance_path = ROOT / STAGE_B_PROVENANCE
    require(summary_path.is_file(), "STAGE_B_SUMMARY_MISSING")
    require(provenance_path.is_file(), "STAGE_B_PROVENANCE_MISSING")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require(
        summary.get("schema_version")
        == "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_B_FUNCTIONAL_DECOMPOSITION_V1",
        "STAGE_B_SCHEMA",
    )
    require(
        summary.get("result")
        == "PASS_GEN5_STAGE_B_FORWARD_FUNCTIONAL_DECOMPOSITION",
        "STAGE_B_RESULT",
    )
    require(summary.get("condition_evaluation_count") == 36, "STAGE_B_CONDITIONS")
    require(summary.get("scientific_conclusion") is None, "STAGE_B_RAW_CONCLUSION")
    require(provenance.get("status") == "PASS", "STAGE_B_PROVENANCE_STATUS")
    require(
        provenance.get("summary_sha256") == sha256_file(summary_path),
        "STAGE_B_SUMMARY_SHA",
    )
    return summary


def _effective_rank_basis(
    value: torch.Tensor,
) -> tuple[int, torch.Tensor, list[float], float]:
    x = value.detach().cpu().to(torch.float64)
    require(x.ndim == 2, "RANK_INPUT_NDIM")
    require(bool(torch.isfinite(x).all()), "RANK_INPUT_NONFINITE")
    u, s, _vh = torch.linalg.svd(x, full_matrices=False)
    singular = [float(v) for v in s.tolist()]
    if not singular or singular[0] == 0.0:
        return 0, u[:, :0], singular, 0.0

    tolerance = max(x.shape) * FLOAT32_EPS * singular[0]
    rank = int(sum(v > tolerance for v in singular))
    return rank, u[:, :rank], singular, float(tolerance)


def subspace_geometry(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, Any]:
    left_rank, ql, left_s, left_tol = _effective_rank_basis(left)
    right_rank, qr, right_s, right_tol = _effective_rank_basis(right)

    result: dict[str, Any] = {
        "left_effective_rank": left_rank,
        "right_effective_rank": right_rank,
        "left_singular_values": left_s,
        "right_singular_values": right_s,
        "left_rank_tolerance": left_tol,
        "right_rank_tolerance": right_tol,
        "rank2_interpretation_valid": left_rank == 2 and right_rank == 2,
        "principal_cosines": [],
        "principal_angles_deg": [],
        "projection_overlap_sum": 0.0,
        "affinity": None,
    }

    if left_rank == 0 or right_rank == 0:
        return result

    cosines = torch.linalg.svdvals(ql.T @ qr).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosines))
    cos_list = [float(v) for v in cosines.tolist()]
    angle_list = [float(v) for v in angles.tolist()]
    overlap = float(torch.sum(cosines * cosines).item())
    denom = min(left_rank, right_rank)

    result.update({
        "principal_cosines": cos_list,
        "principal_angles_deg": angle_list,
        "projection_overlap_sum": overlap,
        "affinity": overlap / denom,
    })
    return result


def projected_energy_fraction(
    value: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    x = value.detach().cpu().to(torch.float64)
    q = basis.detach().cpu().to(torch.float64)
    require(x.ndim == 2, "PROJECTED_VALUE_NDIM")
    require(q.ndim == 2, "PROJECTED_BASIS_NDIM")
    require(x.shape[0] == q.shape[0], "PROJECTED_DIM")
    denom = float(torch.sum(x * x).item())
    require(denom > 0.0 and math.isfinite(denom), "PROJECTED_ZERO_OR_NONFINITE")
    num = float(torch.sum((q.T @ x) ** 2).item())
    return num / denom


def validate_gpu_queues() -> None:
    flat = [cell for gpu in (0, 1) for cell in GPU_QUEUES[gpu]]
    require(len(flat) == 9, "QUEUE_CELL_COUNT")
    require(len(set(flat)) == 9, "QUEUE_DUPLICATE")
    require(set(flat) == set(CELLS), "QUEUE_MATRIX")


def _load_sources() -> tuple[Any, dict[tuple[int, str], dict[str, Any]]]:
    from scripts import (
        reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition
        as stage_b,
    )
    from scripts import train_reason_router_gen5_phase3a_contention as p3a

    artifacts = stage_b.load_phase3a_artifacts()
    static = p3a.validate_static_artifacts()
    require(len(artifacts) == 9, "PHASE3A_CELL_COUNT")
    require(len(static["train_rows"]) == TRAIN_ROWS, "TRAIN_ROWS")
    return static, artifacts


def run_static_verify(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=args.allow_opening_worktree,
    )
    validate_gpu_queues()
    stage_b = validate_stage_b_freeze()
    static, artifacts = _load_sources()
    del artifacts

    print("GEN5_STAGE_C_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"STAGE_B_SCHEMA={stage_b['schema_version']}")
    print(f"TRAIN_ROWS={len(static['train_rows'])}")
    print("CELLS=9")
    print("GPU_WORKERS=2")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")


def _worker_cell(
    *,
    args: argparse.Namespace,
    seed: int,
    pressure: str,
    static: Mapping[str, Any],
    encoded: Mapping[str, Any],
    snapshot: Path,
    checkpoint_path: Path,
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases,
        parent_parameter_fingerprint,
        phase2_final_three_way_ce,
    )
    from scripts import train_reason_router_gen5_phase3a_contention as p3a

    report = artifact["report"]
    require(
        encoded["train_encoding_sha256"] == report["train_encoding_sha256"],
        f"TRAIN_ENCODING:{seed}:{pressure}",
    )

    model, wrapper, runtime_meta, strong_mask, planes = p3a._prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )
    parent_before = runtime_meta["parent_before"]

    features, labels, active, targets = p3a._feature_batch_to_device(
        encoded["train_bundle"],
        torch.device("cuda:0"),
    )

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)

    require(
        int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0,
        f"B_INIT_NONZERO:{seed}:{pressure}",
    )

    output, chunks = p3a._streamed_forward(
        model,
        features,
        active,
        targets,
        pressure=pressure,
        strong_mask=strong_mask,
        planes=planes,
    )
    require(chunks == TRAIN_ROWS // STREAM_ROWS, f"CHUNKS:{seed}:{pressure}:{chunks}")

    logits = output["logits"]
    require(tuple(logits.shape) == (TRAIN_ROWS, 3), "LOGITS_SHAPE")
    loss = phase2_final_three_way_ce(logits, labels)
    require(bool(torch.isfinite(loss).item()), "LOSS_NONFINITE")
    loss.backward()

    a_grad = wrapper.correction.A_theta.weight.grad
    b_grad = wrapper.correction.B_theta.weight.grad
    require(a_grad is not None, f"A_GRAD_MISSING:{seed}:{pressure}")
    require(b_grad is not None, f"B_GRAD_MISSING:{seed}:{pressure}")
    require(bool(torch.isfinite(a_grad).all()), f"A_GRAD_NONFINITE:{seed}:{pressure}")
    require(bool(torch.isfinite(b_grad).all()), f"B_GRAD_NONFINITE:{seed}:{pressure}")

    parent_grads = [
        name
        for name, parameter in model.named_parameters()
        if ".correction." not in name and parameter.grad is not None
    ]
    require(not parent_grads, f"PARENT_GRADIENT:{seed}:{pressure}:{parent_grads[:5]}")
    require(
        parent_parameter_fingerprint(model) == parent_before,
        f"PARENT_MUTATION:{seed}:{pressure}",
    )

    observed_loss = float(loss.detach().cpu().item())
    expected_loss = float(report["step0_loss"])
    loss_delta = abs(observed_loss - expected_loss)
    require(
        loss_delta <= STEP0_LOSS_ATOL,
        f"STEP0_LOSS_REPLAY:{seed}:{pressure}:{observed_loss}:{expected_loss}",
    )

    a_grad_cpu = a_grad.detach().cpu().contiguous()
    g = b_grad.detach().cpu().contiguous()
    require(tuple(g.shape) == (24576, 2), "B_GRAD_SHAPE")
    require(float(torch.linalg.vector_norm(g).item()) > 0.0, "B_GRAD_ZERO")

    a_grad_max_abs = float(torch.max(torch.abs(a_grad_cpu)).item())
    a_grad_nonzero = int(torch.count_nonzero(a_grad_cpu).item())
    require(a_grad_nonzero == 0, f"A_GRAD_NONZERO:{seed}:{pressure}:{a_grad_max_abs}")

    r22, c22, _basis_geometry = load_frozen_owner_bases(ROOT)
    final_b = (
        artifact["payload"]["state_dict"]["B_theta.weight"]
        .detach()
        .cpu()
        .contiguous()
    )

    g_rank, _qg, g_singular, g_rank_tol = _effective_rank_basis(g)
    b_rank, _qb, b_singular, b_rank_tol = _effective_rank_basis(final_b)

    g_vs_r = subspace_geometry(g, r22)
    g_vs_c = subspace_geometry(g, c22)
    g_vs_b = subspace_geometry(g, final_b)

    result = {
        "seed": seed,
        "pressure": pressure,
        "source_phase3a_execution_commit": PHASE3A_EXECUTION_COMMIT,
        "source_final_checkpoint_sha256": artifact["checkpoint_sha256"],
        "train_rows": TRAIN_ROWS,
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "step0_loss_observed": observed_loss,
        "step0_loss_frozen_report": expected_loss,
        "step0_loss_abs_delta": loss_delta,
        "model_mode": "TRAIN",
        "streamed_chunks": chunks,
        "gradient_B_sha256": tensor_sha256(g),
        "gradient_B_frobenius_norm": float(torch.linalg.vector_norm(g).item()),
        "gradient_B_effective_rank": g_rank,
        "gradient_B_singular_values": g_singular,
        "gradient_B_rank_tolerance": g_rank_tol,
        "final_B_effective_rank": b_rank,
        "final_B_singular_values": b_singular,
        "final_B_rank_tolerance": b_rank_tol,
        "A_gradient_nonzero_count": a_grad_nonzero,
        "A_gradient_max_abs": a_grad_max_abs,
        "R22_projected_gradient_energy_fraction":
            projected_energy_fraction(g, r22),
        "C22_projected_gradient_energy_fraction":
            projected_energy_fraction(g, c22),
        "gradient_B_vs_R22": g_vs_r,
        "gradient_B_vs_C22": g_vs_c,
        "gradient_B_vs_final_B": g_vs_b,
        "rank2_random_affinity_expectation": RANK2_RANDOM_AFFINITY,
        "parent_signature_before": parent_before,
        "parent_signature_after": parent_parameter_fingerprint(model),
        "backward_executed": True,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "gradient_clip_executed": False,
        "training_executed": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "runtime": runtime_meta["runtime"],
    }

    torch.cuda.synchronize()
    del output, logits, loss, model, wrapper
    torch.cuda.empty_cache()
    return result, g


def run_worker(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
        execution_authority_commit=args.execution_authority_commit,
    )
    validate_gpu_queues()
    require(args.worker_id in (0, 1), f"WORKER_ID:{args.worker_id}")

    static, artifacts = _load_sources()
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts import train_reason_router_gen5_phase3a_contention as p3a

    snapshot = p2train.resolve_exact_snapshot(args.model_snapshot)
    checkpoint_path = Path(args.checkpoint)
    p3a.validate_checkpoint(checkpoint_path)
    encoded = p3a.load_runtime_encoding(static, args.tokenizer_snapshot)

    queue = GPU_QUEUES[args.worker_id]

    scratch_root = Path(args.scratch_root)
    worker_root = scratch_root / f"worker{args.worker_id}"
    require(not worker_root.exists(), f"WORKER_OUTPUT_COLLISION:{worker_root}")
    worker_root.mkdir(parents=True, exist_ok=False)

    reports: list[dict[str, Any]] = []
    gradients: dict[str, torch.Tensor] = {}

    for seed, pressure in queue:
        report, gradient = _worker_cell(
            args=args,
            seed=seed,
            pressure=pressure,
            static=static,
            encoded=encoded,
            snapshot=snapshot,
            checkpoint_path=checkpoint_path,
            artifact=artifacts[(seed, pressure)],
        )
        reports.append(report)
        gradients[f"seed{seed}_{pressure}"] = gradient

    (worker_root / "worker_reports.json").write_bytes(
        canonical_json_bytes({
            "worker_id": args.worker_id,
            "queue": [[s, p] for s, p in queue],
            "cells": reports,
        })
    )
    torch.save(gradients, worker_root / "worker_gradients.pt")

    print(f"GEN5_STAGE_C_WORKER_PASS worker={args.worker_id} cells={len(reports)}")


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
    return {"device_count": 2, "devices": devices}


def run_matrix(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
        execution_authority_commit=args.execution_authority_commit,
    )
    validate_gpu_queues()
    gpu_meta = _validate_two_t4s()
    validate_stage_b_freeze()
    _static, artifacts = _load_sources()
    del _static, artifacts

    output_root = Path(args.output_root)
    scratch_root = Path(str(output_root) + "_scratch")
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")
    require(not scratch_root.exists(), f"SCRATCH_COLLISION:{scratch_root}")
    scratch_root.mkdir(parents=True, exist_ok=False)

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-worker",
        "--expected-head", args.expected_head,
        "--implementation-freeze-commit", args.implementation_freeze_commit,
        "--execution-authority-commit", args.execution_authority_commit,
        "--model-snapshot", str(args.model_snapshot),
        "--tokenizer-snapshot", str(args.tokenizer_snapshot),
        "--checkpoint", str(args.checkpoint),
        "--scratch-root", str(scratch_root),
    ]

    workers = []
    for worker_id in (0, 1):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        proc = subprocess.Popen(
            common + ["--worker-id", str(worker_id)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        workers.append((worker_id, proc))

    failures = []
    for worker_id, proc in workers:
        stdout, _ = proc.communicate()
        print(f"=== STAGE C WORKER {worker_id} ===")
        print(stdout, end="" if stdout.endswith("\n") else "\n")
        if proc.returncode != 0:
            failures.append((worker_id, proc.returncode))

    require(not failures, f"WORKER_FAILURES:{failures}")

    all_reports: list[dict[str, Any]] = []
    all_gradients: dict[str, torch.Tensor] = {}
    for worker_id in (0, 1):
        worker_root = scratch_root / f"worker{worker_id}"
        payload = json.loads(
            (worker_root / "worker_reports.json").read_text(encoding="utf-8")
        )
        worker_gradients = torch.load(
            worker_root / "worker_gradients.pt",
            map_location="cpu",
            weights_only=True,
        )
        all_reports.extend(payload["cells"])
        for key, value in worker_gradients.items():
            require(key not in all_gradients, f"GRADIENT_DUPLICATE:{key}")
            all_gradients[key] = value

    require(len(all_reports) == 9, f"REPORT_COUNT:{len(all_reports)}")
    require(len(all_gradients) == 9, f"GRADIENT_COUNT:{len(all_gradients)}")
    observed_cells = {(int(r["seed"]), str(r["pressure"])) for r in all_reports}
    require(observed_cells == set(CELLS), "RESULT_MATRIX")

    all_reports.sort(key=lambda r: (int(r["seed"]), PRESSURES.index(str(r["pressure"]))))

    output_root.mkdir(parents=True, exist_ok=False)
    tensor_path = output_root / "step0_gradient_B_tensors.pt"
    torch.save(all_gradients, tensor_path)

    summary = {
        "schema_version":
            "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_C_STEP0_GRADIENT_GEOMETRY_V1",
        "result": "PASS_GEN5_STAGE_C_STEP0_GRADIENT_GEOMETRY",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "stage_b_freeze_commit": STAGE_B_FREEZE_COMMIT,
        "source_phase3a_execution_commit": PHASE3A_EXECUTION_COMMIT,
        "population": "FROZEN_PHASE3A_TRAIN_SPLIT_MATCHED_PRESSURE",
        "train_rows_per_cell": TRAIN_ROWS,
        "cells": all_reports,
        "cell_count": 9,
        "backward_count": 9,
        "forward_count": 9,
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "gpu_queues": {
            str(k): [[s, p] for s, p in v]
            for k, v in GPU_QUEUES.items()
        },
        "gpu_runtime": gpu_meta,
        "gradient_tensor_file_sha256": sha256_file(tensor_path),
        "training_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "gradient_clip_executed": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }

    summary_path = output_root / "step0_gradient_geometry_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary))

    provenance = {
        "schema_version":
            "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_C_PROVENANCE_V1",
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "summary_sha256": sha256_file(summary_path),
        "gradient_tensor_sha256": sha256_file(tensor_path),
        "cell_count": 9,
        "backward_count": 9,
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "training_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "confirmatory_9601_9900_loaded": False,
    }
    provenance_path = output_root / "run_provenance.json"
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    shutil.rmtree(scratch_root)

    print("GEN5_STAGE_C_STEP0_GRADIENT_MATRIX_PASS")
    print("CELLS=9")
    print("FORWARD_COUNT=9")
    print("BACKWARD_COUNT=9")
    print("GPU_WORKERS=2")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("TRAINING_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={summary_path}")
    print(f"GRADIENTS={tensor_path}")
    print(f"PROVENANCE={provenance_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--run-matrix", action="store_true")
    modes.add_argument("--run-worker", action="store_true")

    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--allow-opening-worktree", action="store_true")
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--execution-authority-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--worker-id", type=int)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(args.allow_opening_worktree, "STATIC_OPENING_WORKTREE_REQUIRED")
        for name in (
            "implementation_freeze_commit",
            "execution_authority_commit",
            "model_snapshot",
            "tokenizer_snapshot",
            "checkpoint",
            "output_root",
            "scratch_root",
            "worker_id",
        ):
            require(getattr(args, name) is None, f"STATIC_FORBIDDEN:{name}")
        return

    require(not args.allow_opening_worktree, "RUNTIME_OPENING_WORKTREE_FORBIDDEN")
    for name in (
        "implementation_freeze_commit",
        "execution_authority_commit",
        "model_snapshot",
        "tokenizer_snapshot",
        "checkpoint",
    ):
        require(getattr(args, name) is not None, f"RUNTIME_REQUIRED:{name}")

    if args.run_matrix:
        require(args.output_root is not None, "MATRIX_OUTPUT_ROOT_REQUIRED")
        require(args.scratch_root is None, "MATRIX_SCRATCH_ROOT_FORBIDDEN")
        require(args.worker_id is None, "MATRIX_WORKER_ID_FORBIDDEN")
    else:
        require(args.output_root is None, "WORKER_OUTPUT_ROOT_FORBIDDEN")
        require(args.scratch_root is not None, "WORKER_SCRATCH_ROOT_REQUIRED")
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.static_verify_only:
        run_static_verify(args)
    elif args.run_matrix:
        run_matrix(args)
    else:
        run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
