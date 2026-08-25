from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "pre_urp_latest_resume_checkpoint_v1"
CHECKPOINT_KIND = "LATEST_RESUME"
RESUME_POINT = "AFTER_COMPLETED_EPOCH"
BEST_SCIENTIFIC_CHECKPOINT_KIND = "BEST_SCIENTIFIC_CHECKPOINT"


class ResumeCheckpointError(RuntimeError):
    """Raised when a latest-resume checkpoint is incomplete or mismatched."""


TRUSTED_CHECKPOINT_WARNING = (
    "Latest-resume checkpoints are trusted internal research artifacts. "
    "Loading them uses torch.load(..., weights_only=False), which may execute "
    "pickle deserialization; do not load arbitrary untrusted checkpoint files."
)
DATA_ORDER_CALLER_ESTABLISHED = "CALLER_ESTABLISHED"
DATA_ORDER_NOT_ESTABLISHED = "NOT_ESTABLISHED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResumeCheckpointInfo:
    path: Path
    size_bytes: int
    sha256: str
    completed_epoch: int
    global_optimizer_step: int
    continuation_index: int
    identity: dict[str, Any]
    parent_resume_checkpoint_sha256: str | None
    data_order_exactness: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    cuda_available = bool(include_cuda and torch.cuda.is_available())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "torch_cuda_available_at_capture": cuda_available,
        "torch_cuda_device_count": len(cuda_states) if cuda_states is not None else 0,
    }


def restore_rng_state(
    rng_state: Mapping[str, Any],
    *,
    require_cuda_continuity: bool = False,
) -> dict[str, str]:
    if not isinstance(rng_state, Mapping):
        raise ResumeCheckpointError("rng_state must be a mapping")
    cuda_states = rng_state.get("torch_cuda")
    saved_cuda_device_count = int(rng_state.get("torch_cuda_device_count") or 0)
    cuda_status = "NOT_CHECKED"
    if cuda_states is not None:
        if len(cuda_states) != saved_cuda_device_count:
            raise ResumeCheckpointError(
                "CUDA RNG saved state count mismatch: "
                f"declared={saved_cuda_device_count} states={len(cuda_states)}"
            )
        if not torch.cuda.is_available():
            if require_cuda_continuity:
                raise ResumeCheckpointError("CUDA RNG continuity required but CUDA is unavailable")
            cuda_status = "NOT_CHECKED_CUDA_UNAVAILABLE"
        else:
            current_cuda_device_count = torch.cuda.device_count()
            if saved_cuda_device_count != current_cuda_device_count:
                raise ResumeCheckpointError(
                    "CUDA RNG device count mismatch: "
                    f"saved={saved_cuda_device_count} current={current_cuda_device_count}"
                )
            torch.cuda.set_rng_state_all(cuda_states)
            cuda_status = "RESTORED"
    elif require_cuda_continuity:
        raise ResumeCheckpointError("CUDA RNG continuity required but checkpoint has no CUDA RNG state")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    return {
        "python": "RESTORED",
        "numpy": "RESTORED",
        "torch_cpu": "RESTORED",
        "torch_cuda": cuda_status,
    }


def _torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResumeCheckpointError(f"{name} must be a non-negative integer")
    return value


def _validate_identity(
    identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None,
) -> None:
    if expected_identity is None:
        return
    if not isinstance(identity, Mapping):
        raise ResumeCheckpointError("checkpoint identity metadata is missing")
    for key, expected_value in expected_identity.items():
        if expected_value is None:
            continue
        if key not in identity or identity[key] is None:
            raise ResumeCheckpointError(f"missing required identity field: {key}")
        observed_value = identity[key]
        if type(observed_value) is not type(expected_value) or observed_value != expected_value:
            raise ResumeCheckpointError(
                f"identity mismatch for {key}: "
                f"expected={expected_value!r} ({type(expected_value).__name__}) "
                f"observed={observed_value!r} ({type(observed_value).__name__})"
            )


def _is_portable_value(value: Any, *, allow_arrays: bool = False) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if allow_arrays and (torch.is_tensor(value) or isinstance(value, np.ndarray)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_portable_value(item, allow_arrays=allow_arrays) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, (str, int, float, bool))
            and _is_portable_value(item, allow_arrays=allow_arrays)
            for key, item in value.items()
        )
    return False


def _validate_portable_value(value: Any, name: str, *, allow_arrays: bool = False) -> None:
    if not _is_portable_value(value, allow_arrays=allow_arrays):
        raise ResumeCheckpointError(f"{name} contains a non-portable value")


def _validate_parent_sha(value: Any, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ResumeCheckpointError(
            "parent_resume_checkpoint_sha256 must be a lowercase 64-hex SHA256 string or None"
        )


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_parent_resume_checkpoint_sha256: str | None = None,
) -> None:
    if not isinstance(payload, Mapping):
        raise ResumeCheckpointError("checkpoint payload must be a mapping")
    required = {
        "schema_version",
        "checkpoint_kind",
        "resume_point",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "completed_epoch",
        "global_optimizer_step",
        "rng_state",
        "data_order_state",
        "data_order_state_present",
        "data_order_exactness",
        "best_selection_ledger",
        "identity",
        "parent_resume_checkpoint_sha256",
        "continuation_index",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ResumeCheckpointError(f"missing required checkpoint fields: {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ResumeCheckpointError(
            f"unsupported schema_version={payload['schema_version']!r}"
        )
    if payload["checkpoint_kind"] != CHECKPOINT_KIND:
        raise ResumeCheckpointError(
            f"unsupported checkpoint_kind={payload['checkpoint_kind']!r}"
        )
    if payload["resume_point"] != RESUME_POINT:
        raise ResumeCheckpointError(
            f"unsupported resume_point={payload['resume_point']!r}"
        )
    if not isinstance(payload["model_state_dict"], Mapping):
        raise ResumeCheckpointError("model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise ResumeCheckpointError("optimizer_state_dict must be a mapping")
    _validate_non_negative_int(payload["completed_epoch"], "completed_epoch")
    _validate_non_negative_int(
        payload["global_optimizer_step"], "global_optimizer_step"
    )
    _validate_non_negative_int(payload["continuation_index"], "continuation_index")
    data_order_exactness = payload.get("data_order_exactness")
    if data_order_exactness not in {DATA_ORDER_CALLER_ESTABLISHED, DATA_ORDER_NOT_ESTABLISHED}:
        raise ResumeCheckpointError(
            "data_order_exactness must be CALLER_ESTABLISHED or NOT_ESTABLISHED"
        )
    if not isinstance(payload["data_order_state_present"], bool):
        raise ResumeCheckpointError("data_order_state_present must be a boolean")
    if payload["data_order_state_present"] != (payload["data_order_state"] is not None):
        raise ResumeCheckpointError("data_order_state_present contradicts data_order_state")
    _validate_portable_value(payload["data_order_state"], "data_order_state", allow_arrays=True)
    _validate_portable_value(payload["best_selection_ledger"], "best_selection_ledger")
    _validate_portable_value(payload["identity"], "identity")
    _validate_parent_sha(payload["parent_resume_checkpoint_sha256"])
    _validate_identity(payload["identity"], expected_identity)
    if expected_parent_resume_checkpoint_sha256 is not None:
        observed = payload["parent_resume_checkpoint_sha256"]
        if observed != expected_parent_resume_checkpoint_sha256:
            raise ResumeCheckpointError(
                "parent_resume_checkpoint_sha256 mismatch: "
                f"expected={expected_parent_resume_checkpoint_sha256!r} "
                f"observed={observed!r}"
            )


def _validate_final_parent_chain(payload: Mapping[str, Any], current_sha256: str) -> None:
    parent = payload["parent_resume_checkpoint_sha256"]
    _validate_parent_sha(parent)
    if parent is not None and parent == current_sha256:
        raise ResumeCheckpointError("parent_resume_checkpoint_sha256 cannot equal current checkpoint sha256")


def _fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_latest_resume_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    global_optimizer_step: int,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    rng_state: Mapping[str, Any] | None = None,
    data_order_state: Any | None = None,
    data_order_exactness: str = DATA_ORDER_NOT_ESTABLISHED,
    best_selection_ledger: Any | None = None,
    identity: Mapping[str, Any] | None = None,
    parent_resume_checkpoint_sha256: str | None = None,
    continuation_index: int = 0,
) -> ResumeCheckpointInfo:
    completed_epoch = _validate_non_negative_int(completed_epoch, "completed_epoch")
    global_optimizer_step = _validate_non_negative_int(
        global_optimizer_step, "global_optimizer_step"
    )
    continuation_index = _validate_non_negative_int(
        continuation_index, "continuation_index"
    )
    if data_order_exactness not in {DATA_ORDER_CALLER_ESTABLISHED, DATA_ORDER_NOT_ESTABLISHED}:
        raise ResumeCheckpointError(
            "data_order_exactness must be CALLER_ESTABLISHED or NOT_ESTABLISHED"
        )
    _validate_parent_sha(parent_resume_checkpoint_sha256)
    _validate_portable_value(data_order_state, "data_order_state", allow_arrays=True)
    _validate_portable_value(best_selection_ledger, "best_selection_ledger")
    _validate_portable_value(identity or {}, "identity")
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "resume_point": RESUME_POINT,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "completed_epoch": completed_epoch,
        "global_optimizer_step": global_optimizer_step,
        "rng_state": dict(rng_state or capture_rng_state()),
        "data_order_state": data_order_state,
        "data_order_state_present": data_order_state is not None,
        "data_order_exactness": data_order_exactness,
        "best_selection_ledger": best_selection_ledger,
        "identity": dict(identity or {}),
        "parent_resume_checkpoint_sha256": parent_resume_checkpoint_sha256,
        "continuation_index": continuation_index,
        "latest_resume_semantics": {
            "resume_point": RESUME_POINT,
            "mid_epoch_resume_supported": False,
            "unfinished_batch_resume_supported": False,
            "dataloader_worker_state_saved": False,
            "sampler_position_saved": False,
            "data_order_exactness": data_order_exactness,
            "best_scientific_checkpoint_kind": BEST_SCIENTIFIC_CHECKPOINT_KIND,
            "checkpoint_selection_changed": False,
            "scientific_success_established": False,
        },
    }
    _validate_payload(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=str(checkpoint_path.parent),
            delete=False,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.stat().st_size <= 0:
            raise ResumeCheckpointError("temporary checkpoint is empty")
        os.replace(temporary_path, checkpoint_path)
        temporary_path = None
        _fsync_parent_directory(checkpoint_path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return validate_latest_resume_checkpoint(checkpoint_path)


def validate_latest_resume_checkpoint(
    checkpoint_path: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_parent_resume_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> ResumeCheckpointInfo:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise ResumeCheckpointError(f"checkpoint is missing: {checkpoint_path}")
    if checkpoint_path.stat().st_size <= 0:
        raise ResumeCheckpointError(f"checkpoint is empty: {checkpoint_path}")
    payload = _torch_load(checkpoint_path, map_location=map_location)
    _validate_payload(
        payload,
        expected_identity=expected_identity,
        expected_parent_resume_checkpoint_sha256=(
            expected_parent_resume_checkpoint_sha256
        ),
    )
    current_sha256 = file_sha256(checkpoint_path)
    _validate_final_parent_chain(payload, current_sha256)
    return ResumeCheckpointInfo(
        path=checkpoint_path,
        size_bytes=checkpoint_path.stat().st_size,
        sha256=current_sha256,
        completed_epoch=payload["completed_epoch"],
        global_optimizer_step=payload["global_optimizer_step"],
        continuation_index=payload["continuation_index"],
        identity=dict(payload["identity"]),
        parent_resume_checkpoint_sha256=payload["parent_resume_checkpoint_sha256"],
        data_order_exactness=payload["data_order_exactness"],
    )


def load_latest_resume_checkpoint(
    checkpoint_path: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_parent_resume_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], ResumeCheckpointInfo]:
    info = validate_latest_resume_checkpoint(
        checkpoint_path,
        expected_identity=expected_identity,
        expected_parent_resume_checkpoint_sha256=(
            expected_parent_resume_checkpoint_sha256
        ),
        map_location=map_location,
    )
    payload = _torch_load(info.path, map_location=map_location)
    return dict(payload), info


def restore_latest_resume_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    expected_parent_resume_checkpoint_sha256: str | None = None,
    restore_rng: bool = True,
    require_cuda_rng_continuity: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload, info = load_latest_resume_checkpoint(
        checkpoint_path,
        expected_identity=expected_identity,
        expected_parent_resume_checkpoint_sha256=(
            expected_parent_resume_checkpoint_sha256
        ),
        map_location=map_location,
    )
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_parameter_devices(optimizer)
    scheduler_status = "NONE"
    if payload["scheduler_state_dict"] is not None:
        if scheduler is None:
            raise ResumeCheckpointError("checkpoint contains scheduler state but no scheduler was supplied")
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scheduler_status = "RESTORED"
    scaler_status = "NONE"
    if payload["scaler_state_dict"] is not None:
        if scaler is None:
            raise ResumeCheckpointError("checkpoint contains scaler state but no scaler was supplied")
        scaler.load_state_dict(payload["scaler_state_dict"])
        scaler_status = "RESTORED"
    rng_status = {
        "python": "NOT_CHECKED",
        "numpy": "NOT_CHECKED",
        "torch_cpu": "NOT_CHECKED",
        "torch_cuda": "NOT_CHECKED",
    }
    if restore_rng:
        rng_status = restore_rng_state(
            payload["rng_state"],
            require_cuda_continuity=require_cuda_rng_continuity,
        )
    return {
        "checkpoint_info": info,
        "completed_epoch": payload["completed_epoch"],
        "global_optimizer_step": payload["global_optimizer_step"],
        "continuation_index": payload["continuation_index"],
        "data_order_state": payload["data_order_state"],
        "data_order_state_present": payload["data_order_state_present"],
        "data_order_exactness": payload["data_order_exactness"],
        "best_selection_ledger": payload["best_selection_ledger"],
        "scheduler": scheduler_status,
        "scaler": scaler_status,
        "rng": rng_status,
        "resume_point": payload["resume_point"],
        "trusted_checkpoint_warning": TRUSTED_CHECKPOINT_WARNING,
    }


def _move_optimizer_state_to_parameter_devices(
    optimizer: torch.optim.Optimizer,
) -> None:
    parameter_devices = {
        id(parameter): parameter.device
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    def move_value(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move_value(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [move_value(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item, device) for item in value)
        return value

    def verify_value(value: Any, device: torch.device) -> None:
        if torch.is_tensor(value):
            if value.device != device:
                raise ResumeCheckpointError(
                    f"optimizer state tensor device mismatch: expected={device} observed={value.device}"
                )
        elif isinstance(value, dict):
            for item in value.values():
                verify_value(item, device)
        elif isinstance(value, (list, tuple)):
            for item in value:
                verify_value(item, device)

    for parameter, state in optimizer.state.items():
        device = parameter_devices.get(id(parameter))
        if device is None:
            raise ResumeCheckpointError("optimizer state refers to an unknown parameter")
        optimizer.state[parameter] = move_value(state, device)
        verify_value(optimizer.state[parameter], device)
