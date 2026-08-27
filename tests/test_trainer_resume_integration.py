from __future__ import annotations

import argparse
import copy
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import train_controlled_v6b_minimal as trainer
from scripts.kaggle_resume_store import ResumeStoreError
from scripts.resume_checkpoint import DATA_ORDER_CALLER_ESTABLISHED, DATA_ORDER_NOT_ESTABLISHED
from scripts.trainer_resume import (
    build_trainer_resume_config,
    build_trainer_resume_identity,
    initialize_trainer_resume_state,
    persist_trainer_latest_resume,
    validate_resume_cli_arguments,
)


@pytest.fixture
def work_tmp():
    root = Path.cwd() / ".pre_urp_trainer_resume_test_tmp" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            root.parent.rmdir()
        except OSError:
            pass


def _args(work_tmp: Path, *, store_root: Path | None = None, restore: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        latest_resume_store_root=store_root,
        restore_latest_resume_from_store=restore,
        latest_resume_require_cuda_rng_continuity=False,
        seed=9017,
        resolved_split_seed=42,
        architecture="synthetic",
        backbone="synthetic",
        model_name="synthetic-model",
        data=work_tmp / "synthetic.jsonl",
        reason_router_arm="none",
        fp16=False,
    )


def _model() -> torch.nn.Module:
    return torch.nn.Linear(3, 2)


def _batch():
    x = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    return x, y


def _train_epoch(model, optimizer, scheduler=None):
    x, y = _batch()
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def _run_synthetic_segment(
    *,
    work_tmp: Path,
    store_root: Path | None,
    restore: bool,
    epochs: int,
    trainer_path: Path,
    data_order_exactness: str = DATA_ORDER_NOT_ESTABLISHED,
):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    args = _args(work_tmp, store_root=store_root, restore=restore)
    identity = build_trainer_resume_identity(args, trainer_path=trainer_path, run_name="synthetic")
    config = build_trainer_resume_config(
        args,
        run_dir=work_tmp / ("restored" if restore else "fresh"),
        data_order_exactness=data_order_exactness,
    )
    state = initialize_trainer_resume_state(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_identity=identity,
    )
    step = state.global_optimizer_step
    for epoch in range(state.next_epoch, epochs + 1):
        _train_epoch(model, optimizer, scheduler)
        step += 1
        persist_trainer_latest_resume(
            state=state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            completed_epoch=epoch,
            global_optimizer_step=step,
            identity=identity,
            data_order_state={"synthetic_epoch_order": list(range(1, epochs + 1))},
            best_selection_ledger={"best_epoch": epoch, "best_score": float(epoch)},
        )
    return model, optimizer, scheduler, state, identity


def test_active_trainer_resume_flags_are_default_off():
    parser = trainer.build_parser()
    args = parser.parse_args([])
    assert args.latest_resume_store_root is None
    assert args.restore_latest_resume_from_store is False
    assert args.latest_resume_require_cuda_rng_continuity is False


def test_default_resume_config_has_no_store_access(work_tmp):
    args = _args(work_tmp)
    config = build_trainer_resume_config(args, run_dir=work_tmp)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    state = initialize_trainer_resume_state(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        expected_identity={},
    )
    result = persist_trainer_latest_resume(
        state=state,
        model=model,
        optimizer=optimizer,
        completed_epoch=1,
        global_optimizer_step=1,
        identity={},
    )
    assert result is None
    assert list(work_tmp.iterdir()) == []


def test_fresh_opt_in_persists_latest_resume_and_advances_pointer(work_tmp):
    store_root = work_tmp / "store"
    trainer_path = Path(trainer.__file__)
    model, optimizer, scheduler, state, _ = _run_synthetic_segment(
        work_tmp=work_tmp,
        store_root=store_root,
        restore=False,
        epochs=1,
        trainer_path=trainer_path,
    )
    assert state.last_backup is not None
    assert state.last_backup.completed_epoch == 1
    assert state.last_backup.global_optimizer_step == 1
    assert state.last_backup.continuation_index == 0
    assert (store_root / "latest_resume.json").is_file()
    assert state.last_backup.object_path.is_file()
    assert model is not None and optimizer is not None and scheduler is not None


def test_completed_epoch_resume_matches_uninterrupted_reference(work_tmp):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    reference_model = _model()
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.01)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.9)
    for _ in range(4):
        _train_epoch(reference_model, reference_optimizer, reference_scheduler)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    store_root = work_tmp / "store"
    trainer_path = Path(trainer.__file__)
    interrupted_model, _, _, interrupted_state, _ = _run_synthetic_segment(
        work_tmp=work_tmp,
        store_root=store_root,
        restore=False,
        epochs=2,
        trainer_path=trainer_path,
        data_order_exactness=DATA_ORDER_CALLER_ESTABLISHED,
    )
    parent_sha = interrupted_state.last_backup.sha256
    restored_model, restored_optimizer, restored_scheduler, restored_state, _ = _run_synthetic_segment(
        work_tmp=work_tmp,
        store_root=store_root,
        restore=True,
        epochs=4,
        trainer_path=trainer_path,
        data_order_exactness=DATA_ORDER_CALLER_ESTABLISHED,
    )

    for key, expected in reference_model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[key], expected)
    assert restored_state.restored_checkpoint_info.completed_epoch == 2
    assert restored_state.next_epoch == 3
    assert restored_state.continuation_index == 1
    assert restored_state.last_backup.continuation_index == 1
    assert restored_state.last_backup.global_optimizer_step == 4
    assert restored_state.restored_checkpoint_info.sha256 == parent_sha
    assert restored_state.parent_resume_checkpoint_sha256 == restored_state.last_backup.sha256
    assert interrupted_model is not None
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict()["last_epoch"] == reference_scheduler.state_dict()["last_epoch"]


def test_parent_chain_updates_to_immediately_previous_generation(work_tmp):
    store_root = work_tmp / "store"
    trainer_path = Path(trainer.__file__)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    args = _args(work_tmp, store_root=store_root)
    identity = build_trainer_resume_identity(args, trainer_path=trainer_path, run_name="synthetic")
    config = build_trainer_resume_config(args, run_dir=work_tmp / "run")
    state = initialize_trainer_resume_state(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        expected_identity=identity,
    )

    _train_epoch(model, optimizer)
    first = persist_trainer_latest_resume(
        state=state,
        model=model,
        optimizer=optimizer,
        completed_epoch=1,
        global_optimizer_step=1,
        identity=identity,
    )
    _train_epoch(model, optimizer)
    second = persist_trainer_latest_resume(
        state=state,
        model=model,
        optimizer=optimizer,
        completed_epoch=2,
        global_optimizer_step=2,
        identity=identity,
    )
    assert first.sha256 != second.sha256
    assert state.parent_resume_checkpoint_sha256 == second.sha256
    assert len(list((store_root / "objects").glob("*.pt"))) == 2


def test_restore_failure_prevents_optimizer_step(work_tmp):
    store_root = work_tmp / "store"
    trainer_path = Path(trainer.__file__)
    _run_synthetic_segment(
        work_tmp=work_tmp,
        store_root=store_root,
        restore=False,
        epochs=1,
        trainer_path=trainer_path,
    )
    pointer = store_root / "latest_resume.json"
    pointer.write_text("{not-json", encoding="utf-8")
    model = _model()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    args = _args(work_tmp, store_root=store_root, restore=True)
    identity = build_trainer_resume_identity(args, trainer_path=trainer_path, run_name="synthetic")
    config = build_trainer_resume_config(args, run_dir=work_tmp / "restore")
    with pytest.raises(ResumeStoreError):
        initialize_trainer_resume_state(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            expected_identity=identity,
        )
    for key, expected in before.items():
        assert torch.equal(model.state_dict()[key], expected)


def test_backup_failure_is_not_silently_ignored(work_tmp, monkeypatch):
    store_root = work_tmp / "store"
    trainer_path = Path(trainer.__file__)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    args = _args(work_tmp, store_root=store_root)
    identity = build_trainer_resume_identity(args, trainer_path=trainer_path, run_name="synthetic")
    config = build_trainer_resume_config(args, run_dir=work_tmp / "run")
    state = initialize_trainer_resume_state(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        expected_identity=identity,
    )

    def fail_backup(*args, **kwargs):
        raise ResumeStoreError("synthetic backup failure")

    monkeypatch.setattr("scripts.trainer_resume.backup_latest_resume", fail_backup)
    with pytest.raises(ResumeStoreError):
        persist_trainer_latest_resume(
            state=state,
            model=model,
            optimizer=optimizer,
            completed_epoch=1,
            global_optimizer_step=1,
            identity=identity,
        )
    assert state.last_backup is None


def test_latest_resume_does_not_touch_best_checkpoint_fixture(work_tmp):
    best_checkpoint = work_tmp / "selected_checkpoint.pt"
    best_checkpoint.write_bytes(b"BEST_SCIENTIFIC_CHECKPOINT_PLACEHOLDER")
    before = best_checkpoint.read_bytes()
    _run_synthetic_segment(
        work_tmp=work_tmp,
        store_root=work_tmp / "store",
        restore=False,
        epochs=1,
        trainer_path=Path(trainer.__file__),
    )
    assert best_checkpoint.read_bytes() == before


def test_data_order_exactness_defaults_to_not_established(work_tmp):
    config = build_trainer_resume_config(
        _args(work_tmp, store_root=work_tmp / "store"),
        run_dir=work_tmp / "run",
    )
    assert config.data_order_exactness == DATA_ORDER_NOT_ESTABLISHED


def test_restore_flag_requires_store_root(work_tmp):
    with pytest.raises(ValueError):
        build_trainer_resume_config(_args(work_tmp, restore=True), run_dir=work_tmp)
    with pytest.raises(ValueError):
        validate_resume_cli_arguments(_args(work_tmp, restore=True))
