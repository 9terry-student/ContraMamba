from __future__ import annotations

import json
import os
import shutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import kaggle_resume_store as store
from scripts.resume_checkpoint import (
    DATA_ORDER_CALLER_ESTABLISHED,
    ResumeCheckpointError,
    file_sha256,
    save_latest_resume_checkpoint,
)


NON_SCIENTIFIC_SEED = 9017


@pytest.fixture
def tmp_path(request):
    root = Path.cwd() / ".pre_urp_kaggle_resume_store_test_tmp"
    root.mkdir(exist_ok=True)
    case_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    path = root / case_name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def _model() -> torch.nn.Module:
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    return torch.nn.Linear(2, 2)


def _checkpoint(tmp_path: Path, name: str = "latest_resume.pt", **kwargs) -> Path:
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    model(x).sum().backward()
    optimizer.step()
    path = tmp_path / name
    defaults = {
        "checkpoint_path": path,
        "model": model,
        "optimizer": optimizer,
        "completed_epoch": 1,
        "global_optimizer_step": 1,
        "identity": {
            "run_name": "NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST",
            "seed": NON_SCIENTIFIC_SEED,
        },
        "continuation_index": 1,
    }
    defaults.update(kwargs)
    save_latest_resume_checkpoint(**defaults)
    return path


def _pointer(root: Path) -> dict:
    return json.loads((root / store.POINTER_NAME).read_text(encoding="utf-8"))


def test_status_empty_is_non_error(tmp_path):
    inspection = store.inspect_resume_store(store_root=tmp_path / "store")
    assert inspection.status == "EMPTY"


def test_valid_backup_resolve_and_status(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    result = store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")

    assert result.sha256 == file_sha256(checkpoint)
    assert result.object_path == (tmp_path / "store" / "objects" / f"{result.sha256}.pt").resolve()
    assert result.pointer_path.is_file()
    assert store.resolve_latest_resume(store_root=tmp_path / "store") == result.object_path
    inspection = store.inspect_resume_store(store_root=tmp_path / "store")
    assert inspection.status == "VALID"
    assert inspection.sha256 == result.sha256
    assert inspection.completed_epoch == 1
    assert inspection.global_optimizer_step == 1
    assert inspection.continuation_index == 1


def test_same_checkpoint_backup_is_idempotent(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    first = store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    object_mtime = first.object_path.stat().st_mtime_ns
    second = store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")

    assert second.object_reused is True
    assert second.object_path == first.object_path
    assert second.object_path.stat().st_mtime_ns == object_mtime
    assert len(list((tmp_path / "store" / "objects").glob("*.pt"))) == 1


def test_second_checkpoint_advances_pointer_without_deleting_old_object(tmp_path):
    first_checkpoint = _checkpoint(tmp_path, "first.pt", completed_epoch=1, global_optimizer_step=1, continuation_index=1)
    first = store.backup_latest_resume(first_checkpoint, store_root=tmp_path / "store")
    second_checkpoint = _checkpoint(
        tmp_path,
        "second.pt",
        completed_epoch=2,
        global_optimizer_step=3,
        continuation_index=2,
        parent_resume_checkpoint_sha256=first.sha256,
        data_order_exactness=DATA_ORDER_CALLER_ESTABLISHED,
    )
    second = store.backup_latest_resume(second_checkpoint, store_root=tmp_path / "store")

    assert first.object_path.is_file()
    assert second.object_path.is_file()
    assert first.object_path != second.object_path
    assert store.resolve_latest_resume(store_root=tmp_path / "store") == second.object_path
    assert _pointer(tmp_path / "store")["sha256"] == second.sha256
    assert len(list((tmp_path / "store" / "objects").glob("*.pt"))) == 2


def test_invalid_source_checkpoint_fails_closed(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a valid latest resume checkpoint")
    with pytest.raises(Exception):
        store.backup_latest_resume(bad, store_root=tmp_path / "store")
    assert not (tmp_path / "store" / store.POINTER_NAME).exists()


def test_missing_and_empty_source_fail_closed(tmp_path):
    with pytest.raises(store.ResumeStoreError, match="missing"):
        store.backup_latest_resume(tmp_path / "missing.pt", store_root=tmp_path / "store")
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(store.ResumeStoreError, match="empty"):
        store.backup_latest_resume(empty, store_root=tmp_path / "store")


def test_copy_hash_mismatch_fails_closed_and_cleans_temp(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)

    def bad_copy(_source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupt copied bytes")

    monkeypatch.setattr(store, "_copy_file_atomic", bad_copy)
    with pytest.raises(Exception):
        store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    assert not (tmp_path / "store" / store.POINTER_NAME).exists()
    assert not list((tmp_path / "store" / "objects").glob("*.pt"))


def test_corrupt_pre_existing_sha_object_fails_closed(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    source_sha = file_sha256(checkpoint)
    object_dir = tmp_path / "store" / "objects"
    object_dir.mkdir(parents=True)
    (object_dir / f"{source_sha}.pt").write_bytes(b"corrupt existing object")

    with pytest.raises(Exception):
        store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    assert not (tmp_path / "store" / store.POINTER_NAME).exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("marker", "WRONG", "marker"),
        ("evidence_boundary", "SCIENTIFIC_EVIDENCE", "evidence_boundary"),
        ("checkpoint_kind", "BEST_SCIENTIFIC_CHECKPOINT", "checkpoint_kind"),
        ("resume_point", "MID_EPOCH", "resume_point"),
    ],
)
def test_wrong_pointer_marker_boundary_kind_or_resume_point_fails(tmp_path, field, value, match):
    checkpoint = _checkpoint(tmp_path)
    result = store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    pointer = _pointer(tmp_path / "store")
    pointer[field] = value
    (tmp_path / "store" / store.POINTER_NAME).write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(store.ResumeStoreError, match=match):
        store.resolve_latest_resume(store_root=tmp_path / "store")
    assert store.inspect_resume_store(store_root=tmp_path / "store").status == "INVALID"
    assert result.object_path.is_file()


def test_malformed_pointer_and_missing_object_fail_closed(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    result = store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    (tmp_path / "store" / store.POINTER_NAME).write_text("{", encoding="utf-8")
    with pytest.raises(store.ResumeStoreError, match="malformed JSON"):
        store.resolve_latest_resume(store_root=tmp_path / "store")

    store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    result.object_path.unlink()
    with pytest.raises(store.ResumeStoreError, match="missing"):
        store.resolve_latest_resume(store_root=tmp_path / "store")


@pytest.mark.parametrize("object_filename", ["../x.pt", "/tmp/x.pt", "C:\\tmp\\x.pt", "subdir/x.pt"])
def test_pointer_object_path_traversal_fails(tmp_path, object_filename):
    checkpoint = _checkpoint(tmp_path)
    store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    pointer = _pointer(tmp_path / "store")
    pointer["object_filename"] = object_filename
    (tmp_path / "store" / store.POINTER_NAME).write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(store.ResumeStoreError, match="unsafe"):
        store.resolve_latest_resume(store_root=tmp_path / "store")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("size_bytes", 1, "size mismatch"),
        ("sha256", "0" * 64, "object_filename"),
        ("completed_epoch", 999, "metadata mismatch"),
        ("global_optimizer_step", 999, "metadata mismatch"),
        ("continuation_index", 999, "metadata mismatch"),
        ("identity", {"run_name": "wrong"}, "metadata mismatch"),
        ("data_order_exactness", "CALLER_ESTABLISHED", "metadata mismatch"),
    ],
)
def test_pointer_checkpoint_metadata_mismatch_fails(tmp_path, field, value, match):
    checkpoint = _checkpoint(tmp_path)
    store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
    pointer = _pointer(tmp_path / "store")
    pointer[field] = value
    (tmp_path / "store" / store.POINTER_NAME).write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(store.ResumeStoreError, match=match):
        store.resolve_latest_resume(store_root=tmp_path / "store")


def test_atomic_pointer_replacement_preserves_existing_pointer(tmp_path, monkeypatch):
    first_checkpoint = _checkpoint(tmp_path, "first.pt")
    first = store.backup_latest_resume(first_checkpoint, store_root=tmp_path / "store")
    original_pointer_text = (tmp_path / "store" / store.POINTER_NAME).read_text(encoding="utf-8")
    second_checkpoint = _checkpoint(tmp_path, "second.pt", completed_epoch=2, global_optimizer_step=2)

    real_replace = os.replace

    def failing_replace(source, destination):
        if Path(destination).name == store.POINTER_NAME:
            raise OSError("synthetic pointer replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", failing_replace)
    with pytest.raises(OSError, match="synthetic pointer replace failure"):
        store.backup_latest_resume(second_checkpoint, store_root=tmp_path / "store")

    assert (tmp_path / "store" / store.POINTER_NAME).read_text(encoding="utf-8") == original_pointer_text
    assert store.resolve_latest_resume(store_root=tmp_path / "store") == first.object_path
    assert list((tmp_path / "store").glob(f".{store.POINTER_NAME}.*.tmp")) == []


def test_cli_status_empty_resolve_failure_and_backup_success(tmp_path, capsys):
    assert store.main(["--store-root", str(tmp_path / "store"), "status"]) == 0
    assert "STORE_STATUS: EMPTY" in capsys.readouterr().out
    assert store.main(["--store-root", str(tmp_path / "store"), "resolve"]) == 1
    assert "RESOLVE_STATUS: FAIL" in capsys.readouterr().out

    checkpoint = _checkpoint(tmp_path)
    assert store.main(["--store-root", str(tmp_path / "store"), "backup", "--checkpoint", str(checkpoint)]) == 0
    assert "BACKUP_STATUS: PASS" in capsys.readouterr().out
    assert store.main(["--store-root", str(tmp_path / "store"), "resolve"]) == 0
    assert "RESOLVE_STATUS: PASS" in capsys.readouterr().out


def test_direct_script_status_backup_and_resolve_from_repo_root(tmp_path):
    script = Path("scripts/kaggle_resume_store.py")
    store_root = tmp_path / "direct_cli_store"
    status = subprocess.run(
        [sys.executable, str(script), "--store-root", str(store_root), "status"],
        cwd=Path.cwd(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert status.returncode == 0
    assert "STORE_STATUS: EMPTY" in status.stdout
    assert "ModuleNotFoundError" not in status.stderr

    checkpoint = _checkpoint(tmp_path, "direct_cli_latest.pt")
    backup = subprocess.run(
        [
            sys.executable,
            str(script),
            "--store-root",
            str(store_root),
            "backup",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert backup.returncode == 0
    assert "BACKUP_STATUS: PASS" in backup.stdout
    assert "ModuleNotFoundError" not in backup.stderr

    resolved = subprocess.run(
        [sys.executable, str(script), "--store-root", str(store_root), "resolve"],
        cwd=Path.cwd(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert resolved.returncode == 0
    assert "RESOLVE_STATUS: PASS" in resolved.stdout
    assert "ModuleNotFoundError" not in resolved.stderr


def test_checkpoint_kind_must_remain_latest_resume(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)

    def reject_best(_path):
        raise ResumeCheckpointError("unsupported checkpoint_kind='BEST_SCIENTIFIC_CHECKPOINT'")

    monkeypatch.setattr(store, "validate_latest_resume_checkpoint", reject_best)
    with pytest.raises(ResumeCheckpointError, match="BEST_SCIENTIFIC_CHECKPOINT"):
        store.backup_latest_resume(checkpoint, store_root=tmp_path / "store")
