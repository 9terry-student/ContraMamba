from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.kaggle_resume_store import BackupResult, backup_latest_resume, resolve_latest_resume
from scripts.resume_checkpoint import (
    DATA_ORDER_CALLER_ESTABLISHED,
    DATA_ORDER_NOT_ESTABLISHED,
    RESUME_POINT,
    ResumeCheckpointInfo,
    file_sha256,
    restore_latest_resume_checkpoint,
    save_latest_resume_checkpoint,
)


LATEST_RESUME_FILENAME = "latest_resume.pt"


@dataclass(frozen=True)
class TrainerResumeConfig:
    enabled: bool
    restore_from_store: bool
    store_root: Path | None
    run_local_checkpoint_path: Path | None
    require_cuda_rng_continuity: bool
    data_order_exactness: str = DATA_ORDER_NOT_ESTABLISHED


@dataclass
class TrainerResumeState:
    config: TrainerResumeConfig
    next_epoch: int
    global_optimizer_step: int
    continuation_index: int
    parent_resume_checkpoint_sha256: str | None
    restored_checkpoint_path: Path | None = None
    restored_checkpoint_info: ResumeCheckpointInfo | None = None
    last_backup: BackupResult | None = None


def add_resume_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--latest-resume-store-root",
        type=Path,
        default=None,
        help=(
            "Explicitly enable latest-resume persistence using this persistent "
            "store root. Default disabled."
        ),
    )
    parser.add_argument(
        "--restore-latest-resume-from-store",
        action="store_true",
        default=False,
        help=(
            "Explicitly restore from --latest-resume-store-root before training. "
            "Default starts a fresh execution segment."
        ),
    )
    parser.add_argument(
        "--latest-resume-require-cuda-rng-continuity",
        action="store_true",
        default=False,
        help=(
            "Fail closed unless CUDA RNG state is restored exactly. Default allows "
            "CPU-only restore behavior to remain explicit and non-scientific."
        ),
    )


def validate_resume_cli_arguments(args: argparse.Namespace) -> None:
    if (
        bool(getattr(args, "restore_latest_resume_from_store", False))
        and getattr(args, "latest_resume_store_root", None) is None
    ):
        raise ValueError("--restore-latest-resume-from-store requires --latest-resume-store-root")


def build_trainer_resume_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    data_order_exactness: str = DATA_ORDER_NOT_ESTABLISHED,
) -> TrainerResumeConfig:
    store_root = getattr(args, "latest_resume_store_root", None)
    restore_from_store = bool(getattr(args, "restore_latest_resume_from_store", False))
    enabled = store_root is not None
    validate_resume_cli_arguments(args)
    if data_order_exactness not in {DATA_ORDER_CALLER_ESTABLISHED, DATA_ORDER_NOT_ESTABLISHED}:
        raise ValueError("data_order_exactness must be CALLER_ESTABLISHED or NOT_ESTABLISHED")
    run_local_checkpoint_path = (
        Path(run_dir) / LATEST_RESUME_FILENAME if enabled else None
    )
    return TrainerResumeConfig(
        enabled=enabled,
        restore_from_store=restore_from_store,
        store_root=Path(store_root) if store_root is not None else None,
        run_local_checkpoint_path=run_local_checkpoint_path,
        require_cuda_rng_continuity=bool(
            getattr(args, "latest_resume_require_cuda_rng_continuity", False)
        ),
        data_order_exactness=data_order_exactness,
    )


def build_trainer_resume_identity(
    args: argparse.Namespace,
    *,
    trainer_path: Path,
    run_name: str,
) -> dict[str, Any]:
    resolved_trainer = Path(trainer_path).resolve()
    return {
        "trainer_path": str(resolved_trainer),
        "trainer_sha256": file_sha256(resolved_trainer),
        "run_name": run_name,
        "seed": int(args.seed),
        "split_seed": int(getattr(args, "resolved_split_seed")),
        "architecture": str(args.architecture),
        "backbone": str(args.backbone),
        "model_name": str(args.model_name) if getattr(args, "model_name", None) is not None else None,
        "data_path": str(args.data),
        "reason_router_arm": str(getattr(args, "reason_router_arm", "none")),
        "fp16": bool(getattr(args, "fp16", False)),
        "resume_point": RESUME_POINT,
    }


def initialize_trainer_resume_state(
    *,
    config: TrainerResumeConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    expected_identity: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> TrainerResumeState:
    if not config.enabled:
        return TrainerResumeState(
            config=config,
            next_epoch=1,
            global_optimizer_step=0,
            continuation_index=0,
            parent_resume_checkpoint_sha256=None,
        )
    if config.store_root is None or config.run_local_checkpoint_path is None:
        raise ValueError("enabled latest-resume config requires store and local checkpoint paths")
    if not config.restore_from_store:
        return TrainerResumeState(
            config=config,
            next_epoch=1,
            global_optimizer_step=0,
            continuation_index=0,
            parent_resume_checkpoint_sha256=None,
        )

    checkpoint_path = resolve_latest_resume(store_root=config.store_root)
    restored = restore_latest_resume_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_identity=expected_identity,
        restore_rng=True,
        require_cuda_rng_continuity=config.require_cuda_rng_continuity,
        map_location=map_location,
    )
    info = restored["checkpoint_info"]
    return TrainerResumeState(
        config=config,
        next_epoch=int(restored["completed_epoch"]) + 1,
        global_optimizer_step=int(restored["global_optimizer_step"]),
        continuation_index=int(restored["continuation_index"]) + 1,
        parent_resume_checkpoint_sha256=info.sha256,
        restored_checkpoint_path=checkpoint_path,
        restored_checkpoint_info=info,
    )


def persist_trainer_latest_resume(
    *,
    state: TrainerResumeState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    global_optimizer_step: int,
    identity: Mapping[str, Any],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    data_order_state: Any | None = None,
    best_selection_ledger: Any | None = None,
) -> BackupResult | None:
    config = state.config
    if not config.enabled:
        return None
    if config.store_root is None or config.run_local_checkpoint_path is None:
        raise ValueError("enabled latest-resume config requires store and local checkpoint paths")

    save_info = save_latest_resume_checkpoint(
        checkpoint_path=config.run_local_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        completed_epoch=completed_epoch,
        global_optimizer_step=global_optimizer_step,
        data_order_state=data_order_state,
        data_order_exactness=config.data_order_exactness,
        best_selection_ledger=best_selection_ledger,
        identity=identity,
        parent_resume_checkpoint_sha256=state.parent_resume_checkpoint_sha256,
        continuation_index=state.continuation_index,
    )
    backup = backup_latest_resume(save_info.path, store_root=config.store_root)
    if backup.sha256 != save_info.sha256:
        raise RuntimeError(
            "latest-resume backup SHA mismatch: "
            f"saved={save_info.sha256} backed_up={backup.sha256}"
        )
    state.parent_resume_checkpoint_sha256 = backup.sha256
    state.last_backup = backup
    return backup


def trainer_resume_report(state: TrainerResumeState) -> dict[str, Any]:
    config = state.config
    last_backup = state.last_backup
    return {
        "enabled": config.enabled,
        "restore_requested": config.restore_from_store,
        "resume_point": RESUME_POINT,
        "next_epoch": state.next_epoch,
        "global_optimizer_step": state.global_optimizer_step,
        "continuation_index": state.continuation_index,
        "parent_resume_checkpoint_sha256": state.parent_resume_checkpoint_sha256,
        "data_order_exactness": config.data_order_exactness,
        "restored_checkpoint_sha256": (
            state.restored_checkpoint_info.sha256 if state.restored_checkpoint_info else None
        ),
        "last_persisted_sha256": last_backup.sha256 if last_backup else None,
        "latest_resume_is_best_scientific_checkpoint": False,
        "scientific_success_established": False,
    }
