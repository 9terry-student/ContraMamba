from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.resume_checkpoint import (
    CHECKPOINT_KIND,
    RESUME_POINT,
    TRUSTED_CHECKPOINT_WARNING,
    ResumeCheckpointInfo,
    validate_latest_resume_checkpoint,
)


DEFAULT_STORE_ROOT = Path("/kaggle/working/contramamba_resume_store")
POINTER_NAME = "latest_resume.json"
OBJECTS_DIR = "objects"
STORE_MARKER = "PRE_URP_KAGGLE_RESUME_STORE_METADATA"
EVIDENCE_BOUNDARY = "NOT_SCIENTIFIC_EVIDENCE"


class ResumeStoreError(RuntimeError):
    """Raised when the persistent latest-resume store fails closed."""


@dataclass(frozen=True)
class StoreInspection:
    status: str
    store_root: Path
    latest_checkpoint_path: Path | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    completed_epoch: int | None = None
    global_optimizer_step: int | None = None
    continuation_index: int | None = None
    data_order_exactness: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class BackupResult:
    checkpoint_path: Path
    object_path: Path
    pointer_path: Path
    sha256: str
    size_bytes: int
    object_reused: bool
    completed_epoch: int
    global_optimizer_step: int
    continuation_index: int
    data_order_exactness: str
    trusted_checkpoint_warning: str = TRUSTED_CHECKPOINT_WARNING


def _fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _objects_dir(store_root: Path) -> Path:
    return store_root.resolve() / OBJECTS_DIR


def _pointer_path(store_root: Path) -> Path:
    return store_root.resolve() / POINTER_NAME


def _object_filename(sha256: str) -> str:
    if not _is_sha256(sha256):
        raise ResumeStoreError(f"invalid checkpoint SHA256: {sha256!r}")
    return f"{sha256}.pt"


def _object_path(store_root: Path, sha256: str) -> Path:
    return _objects_dir(store_root) / _object_filename(sha256)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=str(path.parent),
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.stat().st_size <= 0:
            raise ResumeStoreError("temporary pointer file is empty")
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_parent_directory(path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=str(destination.parent),
            delete=False,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.stat().st_size <= 0:
            raise ResumeStoreError("temporary immutable object is empty")
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_parent_directory(destination)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _pointer_payload(info: ResumeCheckpointInfo) -> dict[str, Any]:
    return {
        "marker": STORE_MARKER,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "checkpoint_kind": CHECKPOINT_KIND,
        "resume_point": RESUME_POINT,
        "object_filename": _object_filename(info.sha256),
        "sha256": info.sha256,
        "size_bytes": info.size_bytes,
        "completed_epoch": info.completed_epoch,
        "global_optimizer_step": info.global_optimizer_step,
        "continuation_index": info.continuation_index,
        "identity": info.identity,
        "parent_resume_checkpoint_sha256": info.parent_resume_checkpoint_sha256,
        "data_order_exactness": info.data_order_exactness,
        "trusted_checkpoint_warning": TRUSTED_CHECKPOINT_WARNING,
    }


def _load_pointer(pointer_path: Path) -> dict[str, Any]:
    if not pointer_path.is_file():
        raise ResumeStoreError(f"latest pointer is missing: {pointer_path}")
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResumeStoreError(f"latest pointer is malformed JSON: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResumeStoreError("latest pointer must be a JSON object")
    return payload


def _validate_pointer_schema(pointer: dict[str, Any]) -> None:
    required = {
        "marker",
        "evidence_boundary",
        "checkpoint_kind",
        "resume_point",
        "object_filename",
        "sha256",
        "size_bytes",
        "completed_epoch",
        "global_optimizer_step",
        "continuation_index",
        "identity",
        "parent_resume_checkpoint_sha256",
        "data_order_exactness",
    }
    missing = sorted(required.difference(pointer))
    if missing:
        raise ResumeStoreError(f"latest pointer missing required fields: {missing}")
    if pointer["marker"] != STORE_MARKER:
        raise ResumeStoreError(f"wrong latest pointer marker: {pointer['marker']!r}")
    if pointer["evidence_boundary"] != EVIDENCE_BOUNDARY:
        raise ResumeStoreError(f"wrong latest pointer evidence_boundary: {pointer['evidence_boundary']!r}")
    if pointer["checkpoint_kind"] != CHECKPOINT_KIND:
        raise ResumeStoreError(f"wrong checkpoint_kind: {pointer['checkpoint_kind']!r}")
    if pointer["resume_point"] != RESUME_POINT:
        raise ResumeStoreError(f"wrong resume_point: {pointer['resume_point']!r}")
    if not _is_sha256(pointer["sha256"]):
        raise ResumeStoreError("latest pointer sha256 must be lowercase 64-hex")
    if not isinstance(pointer["size_bytes"], int) or pointer["size_bytes"] <= 0:
        raise ResumeStoreError("latest pointer size_bytes must be a positive integer")
    for field in ("completed_epoch", "global_optimizer_step", "continuation_index"):
        if isinstance(pointer[field], bool) or not isinstance(pointer[field], int) or pointer[field] < 0:
            raise ResumeStoreError(f"latest pointer {field} must be a non-negative integer")
    if not isinstance(pointer["identity"], dict):
        raise ResumeStoreError("latest pointer identity must be an object")


def _safe_pointer_object_path(store_root: Path, pointer: dict[str, Any]) -> Path:
    _validate_pointer_schema(pointer)
    object_filename = pointer["object_filename"]
    expected_filename = _object_filename(pointer["sha256"])
    if not isinstance(object_filename, str):
        raise ResumeStoreError("latest pointer object_filename must be a string")
    pure_windows = PureWindowsPath(object_filename)
    if (
        Path(object_filename).is_absolute()
        or pure_windows.is_absolute()
        or Path(object_filename).name != object_filename
        or pure_windows.name != object_filename
        or "/" in object_filename
        or "\\" in object_filename
        or ":" in object_filename
        or object_filename != expected_filename
    ):
        raise ResumeStoreError("latest pointer object_filename is unsafe or does not match sha256")
    object_path = (_objects_dir(store_root) / object_filename).resolve()
    objects_root = _objects_dir(store_root).resolve()
    try:
        object_path.relative_to(objects_root)
    except ValueError as exc:
        raise ResumeStoreError("latest pointer object path escapes store objects directory") from exc
    return object_path


def _validate_pointer_matches_info(pointer: dict[str, Any], info: ResumeCheckpointInfo) -> None:
    expected = _pointer_payload(info)
    fields = (
        "sha256",
        "size_bytes",
        "completed_epoch",
        "global_optimizer_step",
        "continuation_index",
        "identity",
        "parent_resume_checkpoint_sha256",
        "data_order_exactness",
        "checkpoint_kind",
        "resume_point",
        "object_filename",
    )
    for field in fields:
        if pointer.get(field) != expected[field]:
            raise ResumeStoreError(
                f"latest pointer/checkpoint metadata mismatch for {field}: "
                f"pointer={pointer.get(field)!r} checkpoint={expected[field]!r}"
            )


def _validate_object_bytes(object_path: Path, expected_info: ResumeCheckpointInfo) -> ResumeCheckpointInfo:
    if not object_path.is_file():
        raise ResumeStoreError(f"immutable checkpoint object is missing: {object_path}")
    observed_size = object_path.stat().st_size
    if observed_size != expected_info.size_bytes:
        raise ResumeStoreError(
            f"immutable object size mismatch: expected={expected_info.size_bytes} observed={observed_size}"
        )
    stored_info = validate_latest_resume_checkpoint(object_path)
    if stored_info.sha256 != expected_info.sha256:
        raise ResumeStoreError(
            f"immutable object SHA mismatch: expected={expected_info.sha256} observed={stored_info.sha256}"
        )
    if stored_info.size_bytes != expected_info.size_bytes:
        raise ResumeStoreError(
            f"immutable object validated size mismatch: expected={expected_info.size_bytes} observed={stored_info.size_bytes}"
        )
    return stored_info


def backup_latest_resume(
    checkpoint: Path,
    *,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> BackupResult:
    source = checkpoint.resolve()
    if not source.is_file():
        raise ResumeStoreError(f"source checkpoint is missing: {source}")
    if source.stat().st_size <= 0:
        raise ResumeStoreError(f"source checkpoint is empty: {source}")
    source_info = validate_latest_resume_checkpoint(source)
    object_path = _object_path(store_root, source_info.sha256)
    object_reused = False
    if object_path.exists():
        existing_info = _validate_object_bytes(object_path, source_info)
        _validate_pointer_matches_info(_pointer_payload(source_info), existing_info)
        object_reused = True
    else:
        try:
            _copy_file_atomic(source, object_path)
            copied_size = object_path.stat().st_size if object_path.exists() else -1
            if copied_size != source_info.size_bytes:
                raise ResumeStoreError(
                    f"copied object size mismatch: expected={source_info.size_bytes} observed={copied_size}"
                )
            stored_info = _validate_object_bytes(object_path, source_info)
            if stored_info.sha256 != source_info.sha256:
                raise ResumeStoreError(
                    f"stored object SHA mismatch: expected={source_info.sha256} observed={stored_info.sha256}"
                )
        except Exception:
            try:
                object_path.unlink()
            except FileNotFoundError:
                pass
            raise
    pointer_path = _pointer_path(store_root)
    pointer = _pointer_payload(source_info)
    _atomic_write_json(pointer_path, pointer)
    reloaded_pointer = _load_pointer(pointer_path)
    resolved_info = _resolve_latest_info(store_root, reloaded_pointer)
    if resolved_info.sha256 != source_info.sha256:
        raise ResumeStoreError(
            f"latest pointer resolved wrong SHA: expected={source_info.sha256} observed={resolved_info.sha256}"
        )
    return BackupResult(
        checkpoint_path=source,
        object_path=object_path.resolve(),
        pointer_path=pointer_path.resolve(),
        sha256=source_info.sha256,
        size_bytes=source_info.size_bytes,
        object_reused=object_reused,
        completed_epoch=source_info.completed_epoch,
        global_optimizer_step=source_info.global_optimizer_step,
        continuation_index=source_info.continuation_index,
        data_order_exactness=source_info.data_order_exactness,
    )


def _resolve_latest_info(store_root: Path, pointer: dict[str, Any]) -> ResumeCheckpointInfo:
    object_path = _safe_pointer_object_path(store_root, pointer)
    if not object_path.is_file():
        raise ResumeStoreError(f"latest immutable object is missing: {object_path}")
    if object_path.stat().st_size != pointer["size_bytes"]:
        raise ResumeStoreError(
            f"latest pointer size mismatch: expected={pointer['size_bytes']} observed={object_path.stat().st_size}"
        )
    info = validate_latest_resume_checkpoint(object_path)
    if info.sha256 != pointer["sha256"]:
        raise ResumeStoreError(
            f"latest pointer SHA mismatch: expected={pointer['sha256']} observed={info.sha256}"
        )
    _validate_pointer_matches_info(pointer, info)
    return info


def resolve_latest_resume(
    *,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> Path:
    pointer = _load_pointer(_pointer_path(store_root))
    info = _resolve_latest_info(store_root, pointer)
    return info.path.resolve()


def inspect_resume_store(
    *,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> StoreInspection:
    root = store_root.resolve()
    pointer_path = _pointer_path(root)
    if not pointer_path.exists():
        return StoreInspection(status="EMPTY", store_root=root, detail="latest pointer absent")
    try:
        pointer = _load_pointer(pointer_path)
        info = _resolve_latest_info(root, pointer)
    except Exception as exc:
        return StoreInspection(status="INVALID", store_root=root, detail=str(exc))
    return StoreInspection(
        status="VALID",
        store_root=root,
        latest_checkpoint_path=info.path.resolve(),
        sha256=info.sha256,
        size_bytes=info.size_bytes,
        completed_epoch=info.completed_epoch,
        global_optimizer_step=info.global_optimizer_step,
        continuation_index=info.continuation_index,
        data_order_exactness=info.data_order_exactness,
        detail="latest resume checkpoint validated",
    )


def _print_status(inspection: StoreInspection) -> None:
    print(f"STORE_STATUS: {inspection.status}")
    print(f"store_root: {inspection.store_root}")
    if inspection.latest_checkpoint_path is not None:
        print(f"latest_checkpoint: {inspection.latest_checkpoint_path}")
    if inspection.sha256 is not None:
        print(f"sha256: {inspection.sha256}")
    if inspection.size_bytes is not None:
        print(f"size_bytes: {inspection.size_bytes}")
    if inspection.completed_epoch is not None:
        print(f"completed_epoch: {inspection.completed_epoch}")
    if inspection.global_optimizer_step is not None:
        print(f"global_optimizer_step: {inspection.global_optimizer_step}")
    if inspection.continuation_index is not None:
        print(f"continuation_index: {inspection.continuation_index}")
    if inspection.data_order_exactness is not None:
        print(f"data_order_exactness: {inspection.data_order_exactness}")
    if inspection.detail:
        print(f"detail: {inspection.detail}")
    print(EVIDENCE_BOUNDARY)


def _print_backup(result: BackupResult) -> None:
    print("BACKUP_STATUS: PASS")
    print(f"source_checkpoint: {result.checkpoint_path}")
    print(f"stored_object: {result.object_path}")
    print(f"latest_pointer: {result.pointer_path}")
    print(f"sha256: {result.sha256}")
    print(f"size_bytes: {result.size_bytes}")
    print(f"object_reused: {str(result.object_reused).lower()}")
    print(f"completed_epoch: {result.completed_epoch}")
    print(f"global_optimizer_step: {result.global_optimizer_step}")
    print(f"continuation_index: {result.continuation_index}")
    print(f"data_order_exactness: {result.data_order_exactness}")
    print(EVIDENCE_BOUNDARY)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent pre-URP Kaggle latest-resume checkpoint store."
    )
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    backup = subparsers.add_parser("backup")
    backup.add_argument("--checkpoint", type=Path, required=True)
    subparsers.add_parser("resolve")
    subparsers.add_parser("status")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.operation == "backup":
            _print_backup(backup_latest_resume(args.checkpoint, store_root=args.store_root))
        elif args.operation == "resolve":
            print(f"RESOLVE_STATUS: PASS")
            print(f"checkpoint: {resolve_latest_resume(store_root=args.store_root)}")
            print(EVIDENCE_BOUNDARY)
        elif args.operation == "status":
            inspection = inspect_resume_store(store_root=args.store_root)
            _print_status(inspection)
            return 0 if inspection.status in {"EMPTY", "VALID"} else 1
        else:
            raise ResumeStoreError(f"unknown operation: {args.operation}")
    except Exception as exc:
        print(f"{args.operation.upper()}_STATUS: FAIL")
        print(f"error: {exc}")
        print(EVIDENCE_BOUNDARY)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
