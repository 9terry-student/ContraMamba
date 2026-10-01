#!/usr/bin/env python3
"""Gen5 Stage E fixed-plane causal bottleneck runner.

Current implementation authority:
    33c2c10b8dd14ad4fea826802f3b9163664d2148

Only --static-verify-only is authorized during the implementation phase.
Future CUDA/training modes are fail-closed behind a separate execution
authority and an exact implementation-freeze commit.
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
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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
    ARMS,
    EXPECTED_TRAINABLE_NUMEL,
    fixed_plane_geometry,
    install_stage_e_layer22_wrapper,
    stage_e_active_mask,
    stage_e_optimizer_parameters,
    stage_e_parameter_audit,
)
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

DESIGN_FREEZE_COMMIT = "c2f0c92da999d6508b712eb82f20f33cfcac4624"
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "33c2c10b8dd14ad4fea826802f3b9163664d2148"
)

DESIGN_PATH = (
    "reports/reason_router_gen5_stage_e_"
    "causal_plane_bottleneck_design_candidate.md"
)
IMPLEMENTATION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_"
    "causal_plane_bottleneck_implementation_authority_spec_candidate.md"
)
EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_stage_e_"
    "causal_plane_bottleneck_execution_authority_spec_candidate.md"
)

FROZEN_BLOBS = {
    DESIGN_PATH: "a112a3adfd2faa2b7e141911a4d417d22fa80bfc",
    IMPLEMENTATION_AUTHORITY_PATH:
        "c2f7eddc6052fc0fb203b98164b56c26cbe7a72a",
    "src/contramamba/gen5_phase2_state_update_ownership.py":
        "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "scripts/train_reason_router_gen5_phase2_state_update_ownership.py":
        "8f711776c6b8cab90fbdafcda166780f3295e4ae",
    "scripts/train_reason_router_gen5_phase3a_contention.py":
        "45e333128c4fa31f4f502c5fdcf324243c1084fd",
    "src/contramamba/gen5_phase3_causal_role_contention.py":
        "d8cd9873df23c7cb31f1bbb3c981672295b1af7a",
}

IMPLEMENTATION_PATHS = frozenset({
    "src/contramamba/gen5_stage_e_causal_plane_bottleneck.py",
    "scripts/train_reason_router_gen5_stage_e_causal_plane_bottleneck.py",
    "tests/test_reason_router_gen5_stage_e_causal_plane_bottleneck.py",
})

PREFLIGHT_PREFIX = (
    "reports/reason_router_gen5_stage_e_cuda_preflight_runs/"
)
RUN_PREFIX = (
    "reports/reason_router_gen5_stage_e_causal_plane_bottleneck_runs/"
)

TRAINING_SEEDS = (6201, 6202, 6203)
TRAINING_ARMS = ARMS
PRESSURE = "P0"

TRAIN_ROWS = p3a.TRAIN_ROWS
DEV_ROWS = p3a.DEV_ROWS
TRAIN_PAIRS = p3a.TRAIN_PAIRS
DEV_PAIRS = p3a.DEV_PAIRS
SPLIT_SEED = p3a.SPLIT_SEED
MAX_LENGTH = p3a.MAX_LENGTH
BACKBONE_STREAM_ROWS = p3a.BACKBONE_STREAM_ROWS

TOTAL_OPTIMIZER_STEPS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)
R22_SHA256 = (
    "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
)
C22_SHA256 = (
    "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"
)

ZERO_DEV_CE_P0 = 1.3409655094146729
FREE_P0_GAIN_BY_SEED = {
    6201: 0.4984860420227051,
    6202: 0.5020102858543396,
    6203: 0.4994615912437439,
}
FREE_P0_GAIN_MEAN = 0.4999859730402629

MATRIX_GPU_COUNT = 2
MATRIX_WORKER_CELLS = (
    (
        (6201, "E-R22"),
        (6201, "E-C22"),
        (6203, "E-R22"),
        (6203, "E-C22"),
    ),
    (
        (6202, "E-R22"),
        (6202, "E-C22"),
    ),
)


class StageERunnerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StageERunnerError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StageERunnerError(
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


def validate_checkout_identity(
    branch: str,
    head: str,
    expected_head: str,
) -> str:
    require(head == expected_head, f"HEAD:{head}")
    if branch == EXPECTED_BRANCH:
        return "attached_expected_branch"
    if branch == "":
        return "detached_exact_head"
    raise StageERunnerError(f"BRANCH:{branch}")


def authenticate_repo(
    expected_head: str,
    *,
    allow_opening_worktree: bool = False,
    implementation_freeze_commit: str | None = None,
) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checkout_mode = validate_checkout_identity(
        branch,
        head,
        expected_head,
    )

    for ancestor, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN_FREEZE"),
        (IMPLEMENTATION_AUTHORITY_COMMIT, "IMPLEMENTATION_AUTHORITY"),
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
        require(
            not status,
            f"WORKTREE_NOT_CLEAN:{sorted(status)}",
        )

    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"FROZEN_BLOB:{path}:{observed}",
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
            path
            for path in changed
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
        "checkout_mode": checkout_mode,
        "head": head,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "implementation_authority_commit":
            IMPLEMENTATION_AUTHORITY_COMMIT,
        "implementation_freeze_commit": implementation_freeze_commit,
    }


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"OUTPUT_COLLISION:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


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
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_FIXED_PLANE_SIX_CELL_MATRIX",
        f"IMPLEMENTATION_FREEZE_COMMIT={args.implementation_freeze_commit}",
        "CUDA_PREFLIGHT_ALLOWED=YES",
        "TRAINING_ALLOWED=YES_EXACT_STAGE_E_SIX_CELL",
        "EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV",
        "BACKWARD_ALLOWED=YES_TRAINING_ONLY",
        "OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
    )
    for token in required_tokens:
        require(
            token in text,
            f"EXECUTION_AUTHORITY_TOKEN:{token}",
        )


def _validate_frozen_contract() -> None:
    require(TRAIN_ROWS == 3360, "TRAIN_ROWS")
    require(DEV_ROWS == 840, "DEV_ROWS")
    require(TRAIN_PAIRS == 480, "TRAIN_PAIRS")
    require(DEV_PAIRS == 120, "DEV_PAIRS")
    require(SPLIT_SEED == 16384, "SPLIT_SEED")
    require(TRAINING_SEEDS == (6201, 6202, 6203), "TRAINING_SEEDS")
    require(TRAINING_ARMS == ("E-R22", "E-C22"), "TRAINING_ARMS")
    require(PRESSURE == "P0", "PRESSURE")
    require(TOTAL_OPTIMIZER_STEPS == 20, "OPTIMIZER_STEPS")
    require(LEARNING_RATE == 0.001, "LEARNING_RATE")
    require(WEIGHT_DECAY == 0.0001, "WEIGHT_DECAY")
    require(GRADIENT_CLIP_NORM == 5.0, "GRADIENT_CLIP_NORM")
    require(EXPECTED_TRAINABLE_NUMEL == 1540, "TRAINABLE_NUMEL")
    require(
        math.isclose(
            FREE_P0_GAIN_MEAN,
            sum(FREE_P0_GAIN_BY_SEED.values()) / 3.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "FREE_GAIN_MEAN",
    )


def run_static_verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = authenticate_repo(
        args.expected_head,
        allow_opening_worktree=args.allow_opening_worktree,
    )
    _validate_frozen_contract()

    static = p3a.validate_static_artifacts()
    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)
    del r22, c22

    report = {
        "schema_version": "GEN5_STAGE_E_STATIC_VERIFY_V1",
        "result": "PASS_GEN5_STAGE_E_STATIC_VERIFY",
        "execution_head": args.expected_head,
        "repository": repo,
        "static_tree": static["tree"],
        "train_order_sha256": p3a.row_order_sha256(
            static["train_rows"]
        ),
        "dev_order_sha256": p3a.row_order_sha256(
            static["dev_rows"]
        ),
        "basis_geometry": basis_geometry,
        "arms": list(TRAINING_ARMS),
        "seeds": list(TRAINING_SEEDS),
        "pressure": PRESSURE,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "expected_trainable_numel": EXPECTED_TRAINABLE_NUMEL,
        "zero_dev_ce_p0": ZERO_DEV_CE_P0,
        "free_p0_gain_by_seed": {
            str(k): v for k, v in FREE_P0_GAIN_BY_SEED.items()
        },
        "checkpoint_loaded": False,
        "model_instantiated": False,
        "model_forward_count": 0,
        "cuda_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "training_executed": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
    }

    print("GEN5_STAGE_E_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"STATIC_TREE_SHA256={static['tree']['tree_sha256']}")
    print(f"TRAIN_ROWS={TRAIN_ROWS}")
    print(f"DEV_ROWS={DEV_ROWS}")
    print("ARMS=E-R22,E-C22")
    print("SEEDS=6201,6202,6203")
    print("PRESSURE=P0")
    print("EXPECTED_TRAINABLE_NUMEL=1540")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_INSTANTIATED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("TRAINING_EXECUTED=False")
    print("TASK_EVALUATION_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    return report


def _prepare_runtime_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    static = p3a.validate_static_artifacts()

    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership
        as p2train,
    )

    snapshot = p2train.resolve_exact_snapshot(args.model_snapshot)
    checkpoint_path = Path(args.checkpoint)
    observed_checkpoint = p3a.validate_checkpoint(checkpoint_path)
    require(
        observed_checkpoint == PARENT_CHECKPOINT_SHA256,
        "PARENT_CHECKPOINT_SHA256",
    )
    encoded = p3a.load_runtime_encoding(
        static,
        args.tokenizer_snapshot,
    )
    return static, encoded, snapshot, checkpoint_path


def _prepare_runtime_model(
    *,
    snapshot: Path,
    checkpoint_path: Path,
    seed: int,
    arm: str,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    require(seed in TRAINING_SEEDS, f"SEED:{seed}")
    require(arm in TRAINING_ARMS, f"ARM:{arm}")

    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership
        as p2train,
    )

    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    kernels = kernel_compat.load_exact_fast_kernels()
    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)

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
    wrapper = install_stage_e_layer22_wrapper(
        model,
        arm=arm,
        r22=r22,
        c22=c22,
        seed=seed,
    )
    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_INSTALL_MUTATION",
    )
    require(
        int(torch.count_nonzero(
            wrapper.correction.M_theta.weight
        ).item()) == 0,
        "M_NOT_ZERO_INITIALIZED",
    )

    audit = stage_e_parameter_audit(model)
    require(
        audit["trainable_tensor_count"] == 2,
        "TRAINABLE_TENSOR_COUNT",
    )
    require(
        audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL,
        "TRAINABLE_NUMEL",
    )

    return model, wrapper, {
        "runtime": runtime,
        "constructor_counts": constructor_counts,
        "parent_before": parent_before,
        "basis_geometry": basis_geometry,
        "parameter_audit": audit,
    }


def _bundle_to_device(
    bundle: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = bundle["model_inputs"]
    features = {
        key: inputs[key].to(device)
        for key in (
            "input_ids",
            "attention_mask",
            "claim_mask",
            "evidence_mask",
        )
    }
    labels = inputs["final_labels"].to(device)
    return features, labels


def _historical_forward_from_hidden(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    hidden_states: torch.Tensor,
) -> Mapping[str, Any]:
    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership
        as p2train,
    )
    return p2train._historical_forward_from_hidden(
        model,
        features,
        hidden_states,
    )


def _streamed_backbone_hidden(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    *,
    stream_rows: int = BACKBONE_STREAM_ROWS,
) -> tuple[torch.Tensor, int]:
    from scripts import (
        train_reason_router_gen5_phase2_state_update_ownership
        as p2train,
    )

    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    require(
        tuple(attention_mask.shape) == tuple(input_ids.shape),
        "ATTENTION_SHAPE",
    )
    row_count = int(input_ids.shape[0])

    rng_before = p2train._capture_rng_state()
    chunks: list[torch.Tensor] = []

    for start in range(0, row_count, stream_rows):
        stop = min(start + stream_rows, row_count)

        def chunk_forward(
            chunk_input_ids: torch.Tensor,
            chunk_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            with stage_e_active_mask(
                model,
                chunk_attention_mask,
            ):
                result = model.mamba(
                    input_ids=chunk_input_ids
                )
            return result.last_hidden_state

        chunk_hidden = torch_checkpoint(
            chunk_forward,
            input_ids[start:stop],
            attention_mask[start:stop],
            use_reentrant=False,
            preserve_rng_state=True,
        )
        chunks.append(chunk_hidden)

    rng_after = p2train._capture_rng_state()
    p2train._require_rng_state_equal(
        rng_before,
        rng_after,
        "STAGE_E_STREAM",
    )

    hidden = torch.cat(chunks, dim=0)
    require(
        int(hidden.shape[0]) == row_count,
        "STREAM_ROW_COUNT",
    )
    return hidden, len(chunks)


def _streamed_forward(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
) -> tuple[Mapping[str, Any], int]:
    hidden, chunks = _streamed_backbone_hidden(
        model,
        features,
    )
    return (
        _historical_forward_from_hidden(
            model,
            features,
            hidden,
        ),
        chunks,
    )


def run_cuda_preflight(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=
            args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()
    require(args.seed in TRAINING_SEEDS, "PREFLIGHT_SEED")
    require(args.arm in TRAINING_ARMS, "PREFLIGHT_ARM")
    require(
        args.preflight_output is not None,
        "PREFLIGHT_OUTPUT_REQUIRED",
    )

    static, encoded, snapshot, checkpoint_path = (
        _prepare_runtime_inputs(args)
    )
    model, wrapper, runtime_meta = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        arm=args.arm,
    )
    features, labels = _bundle_to_device(
        encoded["train_bundle"],
        torch.device("cuda:0"),
    )

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model.zero_grad(set_to_none=True)

    output, chunks = _streamed_forward(model, features)
    logits = output["logits"]
    require(
        tuple(logits.shape) == (TRAIN_ROWS, 3),
        "PREFLIGHT_LOGIT_SHAPE",
    )
    require(
        bool(torch.isfinite(logits).all().item()),
        "PREFLIGHT_LOGIT_FINITE",
    )

    loss = phase2_final_three_way_ce(logits, labels)
    require(bool(torch.isfinite(loss).item()), "PREFLIGHT_LOSS")
    loss.backward()

    a_grad = wrapper.correction.A_theta.weight.grad
    m_grad = wrapper.correction.M_theta.weight.grad
    require(a_grad is not None, "A_GRAD_MISSING")
    require(m_grad is not None, "M_GRAD_MISSING")
    require(
        bool(torch.isfinite(a_grad).all().item()),
        "A_GRAD_NONFINITE",
    )
    require(
        bool(torch.isfinite(m_grad).all().item()),
        "M_GRAD_NONFINITE",
    )

    parent_grads = [
        name
        for name, parameter in model.named_parameters()
        if ".correction." not in name and parameter.grad is not None
    ]
    require(
        not parent_grads,
        f"PARENT_GRADIENT:{parent_grads[:5]}",
    )

    geometry = fixed_plane_geometry(wrapper.correction)

    report = {
        "schema_version": "GEN5_STAGE_E_CUDA_PREFLIGHT_V1",
        "result": "PASS_GEN5_STAGE_E_CUDA_PREFLIGHT",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "seed": args.seed,
        "arm": args.arm,
        "pressure": PRESSURE,
        "train_rows": TRAIN_ROWS,
        "stream_chunk_count": chunks,
        "runtime": runtime_meta,
        "train_order_sha256": p3a.row_order_sha256(
            static["train_rows"]
        ),
        "train_encoding_sha256":
            encoded["train_encoding_sha256"],
        "loss": float(loss.detach().cpu().item()),
        "a_gradient_norm": float(
            torch.linalg.vector_norm(a_grad).detach().cpu().item()
        ),
        "m_gradient_norm": float(
            torch.linalg.vector_norm(m_grad).detach().cpu().item()
        ),
        "geometry": geometry,
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

    print("GEN5_STAGE_E_CUDA_PREFLIGHT_PASS")
    print(f"SEED={args.seed}")
    print(f"ARM={args.arm}")
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
) -> dict[str, Any]:
    a = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    m = wrapper.correction.M_theta.weight.detach().cpu().contiguous()

    return {
        "schema_version": "GEN5_STAGE_E_FINAL_CORRECTION_V1",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "seed": args.seed,
        "arm": args.arm,
        "pressure": PRESSURE,
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
        implementation_freeze_commit=
            args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()

    require(args.seed in TRAINING_SEEDS, f"SEED:{args.seed}")
    require(args.arm in TRAINING_ARMS, f"ARM:{args.arm}")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")

    static, encoded, snapshot, checkpoint_path = (
        _prepare_runtime_inputs(args)
    )

    cell_dir = (
        Path(args.output_root)
        / f"seed{args.seed}"
        / args.arm
    )
    require(
        not cell_dir.exists(),
        f"CELL_OUTPUT_COLLISION:{cell_dir}",
    )
    cell_dir.mkdir(parents=True, exist_ok=False)

    model, wrapper, runtime_meta = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        arm=args.arm,
    )
    parent_before = runtime_meta["parent_before"]

    train_features, train_labels = _bundle_to_device(
        encoded["train_bundle"],
        torch.device("cuda:0"),
    )
    dev_features, dev_labels = _bundle_to_device(
        encoded["dev_bundle"],
        torch.device("cuda:0"),
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
        train_reason_router_gen5_phase2_state_update_ownership
        as p2train,
    )

    step0_rng_state = p2train._capture_rng_state()
    losses: list[float] = []
    grad_norms: list[float] = []

    for step in range(TOTAL_OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        output, chunks = _streamed_forward(
            model,
            train_features,
        )
        require(
            chunks
            == math.ceil(TRAIN_ROWS / BACKBONE_STREAM_ROWS),
            f"CHUNKS:{chunks}",
        )

        logits = output["logits"]
        require(
            tuple(logits.shape) == (TRAIN_ROWS, 3),
            "TRAIN_LOGIT_SHAPE",
        )
        require(
            bool(torch.isfinite(logits).all().item()),
            f"TRAIN_LOGIT_NONFINITE:{step}",
        )

        loss = phase2_final_three_way_ce(
            logits,
            train_labels,
        )
        require(
            bool(torch.isfinite(loss).item()),
            f"LOSS_NONFINITE:{step}",
        )
        loss.backward()

        a_grad = wrapper.correction.A_theta.weight.grad
        m_grad = wrapper.correction.M_theta.weight.grad
        require(
            a_grad is not None and m_grad is not None,
            f"GRAD_MISSING:{step}",
        )
        require(
            bool(torch.isfinite(a_grad).all().item())
            and bool(torch.isfinite(m_grad).all().item()),
            f"GRAD_NONFINITE:{step}",
        )

        parent_grads = [
            name
            for name, parameter in model.named_parameters()
            if ".correction." not in name
            and parameter.grad is not None
        ]
        require(
            not parent_grads,
            f"PARENT_GRADIENT:{parent_grads[:5]}",
        )

        clipped = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters,
            GRADIENT_CLIP_NORM,
        )
        require(
            bool(torch.isfinite(clipped).item()),
            "GRAD_NORM_NONFINITE",
        )
        optimizer.step()

        require(
            bool(torch.isfinite(
                wrapper.correction.A_theta.weight
            ).all().item())
            and bool(torch.isfinite(
                wrapper.correction.M_theta.weight
            ).all().item()),
            "CORRECTION_PARAMETER_NONFINITE",
        )

        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(float(clipped.detach().cpu().item()))
        del output, logits, loss
        torch.cuda.synchronize()

    require(
        len(losses) == TOTAL_OPTIMIZER_STEPS,
        "LOSS_COUNT",
    )
    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_MUTATION_AFTER_TRAINING",
    )

    post_training_rng_state = p2train._capture_rng_state()
    optimizer.zero_grad(set_to_none=True)
    p2train._restore_rng_state(step0_rng_state)

    post_output, post_chunks = _streamed_forward(
        model,
        train_features,
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
        "STAGE_E_POST_STEP20_DIAGNOSTIC_RESTORE",
    )

    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_MUTATION_AFTER_POST_STEP20",
    )

    model.eval()
    with torch.no_grad():
        dev_output, dev_chunks = _streamed_forward(
            model,
            dev_features,
        )
        dev_logits = dev_output["logits"]
        require(
            tuple(dev_logits.shape) == (DEV_ROWS, 3),
            "DEV_LOGIT_SHAPE",
        )
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

    gain = ZERO_DEV_CE_P0 - dev_ce
    free_gain = FREE_P0_GAIN_BY_SEED[args.seed]
    recovery = gain / free_gain

    geometry = fixed_plane_geometry(wrapper.correction)
    require(
        float(geometry["fixed_plane_residual_max_abs"]) <= 5e-6,
        "FIXED_PLANE_RESIDUAL",
    )

    checkpoint_out = cell_dir / "final_correction.pt"
    torch.save(
        _checkpoint_payload(wrapper=wrapper, args=args),
        checkpoint_out,
    )

    audit = stage_e_parameter_audit(model)
    a_tensor = wrapper.correction.A_theta.weight.detach().cpu()
    m_tensor = wrapper.correction.M_theta.weight.detach().cpu()

    report = {
        "schema_version": "GEN5_STAGE_E_TRAINING_REPORT_V1",
        "result": "PASS_GEN5_STAGE_E_TRAINING_CELL",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "seed": args.seed,
        "arm": args.arm,
        "selected_basis": (
            "R22" if args.arm == "E-R22" else "C22"
        ),
        "pressure": PRESSURE,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "train_order_sha256": p3a.row_order_sha256(
            static["train_rows"]
        ),
        "dev_order_sha256": p3a.row_order_sha256(
            static["dev_rows"]
        ),
        "train_encoding_sha256":
            encoded["train_encoding_sha256"],
        "dev_encoding_sha256":
            encoded["dev_encoding_sha256"],
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
        "fixed_plane_geometry": geometry,
        "trainable_tensor_names": audit["trainable_names"],
        "trainable_tensor_count": audit["trainable_tensor_count"],
        "trainable_numel": audit["trainable_numel"],
        "A_theta_sha256": tensor_sha256(a_tensor),
        "M_theta_sha256": tensor_sha256(m_tensor),
        "parent_signature_before": parent_before,
        "parent_signature_after":
            parent_parameter_fingerprint(model),
        "runtime": runtime_meta["runtime"],
        "train_stream_chunks":
            math.ceil(TRAIN_ROWS / BACKBONE_STREAM_ROWS),
        "post_step20_stream_chunks": post_chunks,
        "dev_stream_chunks": dev_chunks,
        "final_correction_file_sha256":
            sha256_file(checkpoint_out),
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "optimizer_step_count": TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(
        cell_dir / "training_report.json",
        report,
    )

    provenance = {
        "schema_version": "GEN5_STAGE_E_TRAINING_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "seed": args.seed,
        "arm": args.arm,
        "pressure": PRESSURE,
        "training_report_sha256":
            sha256_file(cell_dir / "training_report.json"),
        "final_correction_file_sha256":
            sha256_file(checkpoint_out),
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "optimizer_step_count": TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(
        cell_dir / "run_provenance.json",
        provenance,
    )

    print(
        f"GEN5_STAGE_E_CELL_PASS "
        f"seed={args.seed} arm={args.arm} "
        f"dev_ce={dev_ce:.12g} "
        f"gain={gain:.12g} "
        f"recovery={recovery:.12g}"
    )
    return report


def _validate_two_t4s() -> dict[str, Any]:
    require(
        torch.cuda.is_available(),
        "CUDA_UNAVAILABLE",
    )
    require(
        torch.cuda.device_count() == MATRIX_GPU_COUNT,
        f"GPU_COUNT:{torch.cuda.device_count()}",
    )

    rows = []
    for index in range(MATRIX_GPU_COUNT):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        require(name == "Tesla T4", f"GPU_NAME:{index}:{name}")
        require(
            capability == (7, 5),
            f"GPU_CAPABILITY:{index}:{capability}",
        )
        rows.append({
            "index": index,
            "name": name,
            "capability": list(capability),
        })
    return {"gpu_count": MATRIX_GPU_COUNT, "devices": rows}


def _cell_command(
    args: argparse.Namespace,
    *,
    seed: int,
    arm: str,
    scratch_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-cell",
        "--expected-head",
        args.expected_head,
        "--implementation-freeze-commit",
        str(args.implementation_freeze_commit),
        "--execution-authority-commit",
        str(args.execution_authority_commit),
        "--seed",
        str(seed),
        "--arm",
        arm,
        "--checkpoint",
        str(args.checkpoint),
        "--output-root",
        str(scratch_root),
    ]
    if args.model_snapshot is not None:
        command += [
            "--model-snapshot",
            str(args.model_snapshot),
        ]
    if args.tokenizer_snapshot is not None:
        command += [
            "--tokenizer-snapshot",
            str(args.tokenizer_snapshot),
        ]
    return command


def _run_worker(
    args: argparse.Namespace,
    *,
    worker_index: int,
    cells: Sequence[tuple[int, str]],
    scratch_root: Path,
    log_path: Path,
) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(worker_index)

    with log_path.open("w", encoding="utf-8") as log:
        for seed, arm in cells:
            command = _cell_command(
                args,
                seed=seed,
                arm=arm,
                scratch_root=scratch_root,
            )
            log.write(
                "COMMAND=" + " ".join(command) + "\n"
            )
            log.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                return int(completed.returncode)
    return 0


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        implementation_freeze_commit=
            args.implementation_freeze_commit,
    )
    _validate_execution_authority(args)
    _validate_frozen_contract()
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")

    gpu_meta = _validate_two_t4s()
    output_root = Path(args.output_root)
    require(
        not output_root.exists(),
        f"OUTPUT_COLLISION:{output_root}",
    )

    scratch_parent = Path(
        tempfile.mkdtemp(prefix="contramamba_stage_e_")
    )
    worker_roots = [
        scratch_parent / f"worker{i}"
        for i in range(MATRIX_GPU_COUNT)
    ]
    worker_logs = [
        scratch_parent / f"worker{i}.log"
        for i in range(MATRIX_GPU_COUNT)
    ]

    args_payload = {
        key: (
            str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }

    processes = []
    for worker_index, cells in enumerate(MATRIX_WORKER_CELLS):
        worker_roots[worker_index].mkdir(parents=True)
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from scripts."
                "train_reason_router_gen5_stage_e_causal_plane_bottleneck "
                "import _matrix_worker_entry; "
                "raise SystemExit(_matrix_worker_entry("
                f"{worker_index!r}, "
                f"{list(cells)!r}, "
                f"{str(worker_roots[worker_index])!r}, "
                f"{str(worker_logs[worker_index])!r}, "
                f"{args_payload!r}"
                "))"
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

    return_codes = [p.wait() for p in processes]

    if any(code != 0 for code in return_codes):
        output_root.mkdir(parents=True, exist_ok=False)
        fail_dir = output_root / "failed_worker_logs"
        fail_dir.mkdir()
        for src in worker_logs:
            if src.exists():
                shutil.copy2(src, fail_dir / src.name)
        raise StageERunnerError(
            f"MATRIX_WORKER_FAILURE:{return_codes}"
        )

    output_root.mkdir(parents=True, exist_ok=False)

    cell_reports = []
    for worker_index, worker_root in enumerate(worker_roots):
        for seed, arm in MATRIX_WORKER_CELLS[worker_index]:
            source = worker_root / f"seed{seed}" / arm
            require(source.is_dir(), f"WORKER_CELL_MISSING:{seed}:{arm}")
            destination = output_root / f"seed{seed}" / arm
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)

            report = json.loads(
                (destination / "training_report.json")
                .read_text(encoding="utf-8")
            )
            cell_reports.append(report)

    cell_reports.sort(key=lambda x: (int(x["seed"]), str(x["arm"])))
    require(len(cell_reports) == 6, "MATRIX_CELL_COUNT")

    by_seed = {}
    for seed in TRAINING_SEEDS:
        seed_rows = {
            str(row["arm"]): row
            for row in cell_reports
            if int(row["seed"]) == seed
        }
        require(
            set(seed_rows) == set(TRAINING_ARMS),
            f"MATRIX_SEED_ARMS:{seed}",
        )
        by_seed[str(seed)] = {
            "E-R22_recovery":
                seed_rows["E-R22"]["recovery_vs_free_p0"],
            "E-C22_recovery":
                seed_rows["E-C22"]["recovery_vs_free_p0"],
            "delta_R22_minus_C22":
                seed_rows["E-R22"]["recovery_vs_free_p0"]
                - seed_rows["E-C22"]["recovery_vs_free_p0"],
        }

    mean_r = sum(
        float(row["recovery_vs_free_p0"])
        for row in cell_reports
        if row["arm"] == "E-R22"
    ) / 3.0
    mean_c = sum(
        float(row["recovery_vs_free_p0"])
        for row in cell_reports
        if row["arm"] == "E-C22"
    ) / 3.0

    summary = {
        "schema_version": "GEN5_STAGE_E_MATRIX_SUMMARY_V1",
        "result": "PASS_GEN5_STAGE_E_SIX_CELL_MATRIX",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "gpu_topology":
            "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "gpu_meta": gpu_meta,
        "worker_cells": [
            [[seed, arm] for seed, arm in cells]
            for cells in MATRIX_WORKER_CELLS
        ],
        "cell_count": 6,
        "cells": cell_reports,
        "seed_matched_recovery": by_seed,
        "mean_recovery_E_R22": mean_r,
        "mean_recovery_E_C22": mean_c,
        "mean_delta_R22_minus_C22": mean_r - mean_c,
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "optimizer_step_count": 6 * TOTAL_OPTIMIZER_STEPS,
        "task_evaluation_executed": True,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    _write_json_once(
        output_root / "matrix_summary.json",
        summary,
    )

    provenance = {
        "schema_version": "GEN5_STAGE_E_MATRIX_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "matrix_summary_sha256":
            sha256_file(output_root / "matrix_summary.json"),
        "cell_count": 6,
        "training_executed": True,
        "backward_executed": True,
        "optimizer_constructed": True,
        "optimizer_step_count": 6 * TOTAL_OPTIMIZER_STEPS,
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

    print("GEN5_STAGE_E_SIX_CELL_MATRIX_PASS")
    print("CELLS=6")
    print("GPU_WORKERS=2")
    print(
        "GPU_TOPOLOGY="
        "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"
    )
    print("PRESSURE=P0")
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_OPTIMIZER_STEPS=120")
    print("TRAINING_EXECUTED=True")
    print("BACKWARD_EXECUTED=True")
    print("TASK_EVALUATION_EXECUTED=True")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={output_root / 'matrix_summary.json'}")
    print(f"PROVENANCE={output_root / 'run_provenance.json'}")
    return summary


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
    cells: Sequence[tuple[int, str]],
    scratch_root: str,
    log_path: str,
    args_dict: Mapping[str, Any],
) -> int:
    args = _namespace_from_dict(args_dict)
    return _run_worker(
        args,
        worker_index=worker_index,
        cells=cells,
        scratch_root=Path(scratch_root),
        log_path=Path(log_path),
    )


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
    parser.add_argument("--arm", choices=TRAINING_ARMS)
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
        require(args.arm is None, "STATIC_ARM_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(
            args.preflight_output is None,
            "STATIC_PREFLIGHT_OUTPUT_FORBIDDEN",
        )
        require(
            args.output_root is None,
            "STATIC_OUTPUT_ROOT_FORBIDDEN",
        )
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
    require(
        args.checkpoint is not None,
        "RUNTIME_CHECKPOINT_REQUIRED",
    )

    if args.cuda_preflight_only:
        require(args.seed in TRAINING_SEEDS, "PREFLIGHT_SEED_REQUIRED")
        require(args.arm in TRAINING_ARMS, "PREFLIGHT_ARM_REQUIRED")
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
        require(args.arm in TRAINING_ARMS, "RUN_CELL_ARM_REQUIRED")
        require(
            args.output_root is not None,
            "RUN_CELL_OUTPUT_ROOT_REQUIRED",
        )
        require(
            args.preflight_output is None,
            "RUN_CELL_PREFLIGHT_OUTPUT_FORBIDDEN",
        )
    elif args.run_matrix:
        require(args.seed is None, "MATRIX_SEED_FORBIDDEN")
        require(args.arm is None, "MATRIX_ARM_FORBIDDEN")
        require(
            args.output_root is not None,
            "MATRIX_OUTPUT_ROOT_REQUIRED",
        )
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
