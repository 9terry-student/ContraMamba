from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch
import scripts.resume_checkpoint as resume_checkpoint

from scripts.resume_checkpoint import (
    DATA_ORDER_CALLER_ESTABLISHED,
    DATA_ORDER_NOT_ESTABLISHED,
    RESUME_POINT,
    TRUSTED_CHECKPOINT_WARNING,
    ResumeCheckpointError,
    capture_rng_state,
    file_sha256,
    load_latest_resume_checkpoint,
    restore_latest_resume_checkpoint,
    restore_rng_state,
    save_latest_resume_checkpoint,
    validate_latest_resume_checkpoint,
)


NON_SCIENTIFIC_SEED = 9017
PARENT_SHA = "a" * 64
OTHER_PARENT_SHA = "b" * 64
EXPECTED_HASH = "c" * 64


class MockScaler:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = state["scale"]


class CustomPayload:
    pass


def _model() -> torch.nn.Module:
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    )


def _advance_once(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    y = model(x).sum()
    y.backward()
    optimizer.step()
    optimizer.zero_grad()


def _assert_state_dict_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def _assert_nested_state_equal(left, right) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left)
        assert torch.is_tensor(right)
        assert torch.equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict)
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_state_equal(left_item, right_item)
    else:
        assert left == right


def _base_checkpoint(tmp_path, **overrides):
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    _advance_once(model, optimizer)
    kwargs = {
        "checkpoint_path": tmp_path / "latest_resume.pt",
        "model": model,
        "optimizer": optimizer,
        "completed_epoch": 1,
        "global_optimizer_step": 1,
    }
    kwargs.update(overrides)
    return save_latest_resume_checkpoint(**kwargs)


def test_latest_resume_save_restore_full_epoch_boundary_state(tmp_path) -> None:
    random.seed(NON_SCIENTIFIC_SEED)
    np.random.seed(NON_SCIENTIFIC_SEED)
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = MockScaler(7.0)
    _advance_once(model, optimizer)
    scheduler.step()
    expected_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    random.seed(NON_SCIENTIFIC_SEED)
    np.random.seed(NON_SCIENTIFIC_SEED)
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    expected_rng_next = (
        random.random(),
        float(np.random.rand()),
        torch.rand(1),
    )
    random.seed(NON_SCIENTIFIC_SEED)
    np.random.seed(NON_SCIENTIFIC_SEED)
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    identity = {
        "repository_head": "synthetic-head",
        "trainer_sha256": "synthetic-trainer-sha",
        "run_name": "NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST",
        "seed": NON_SCIENTIFIC_SEED,
        "split_seed": 77,
        "controlled_dataset_sha256": EXPECTED_HASH,
        "sidecar_sha256": "synthetic-sidecar-sha",
        "provenance_sha256": "synthetic-provenance-sha",
    }
    info = save_latest_resume_checkpoint(
        checkpoint_path=tmp_path / "latest_resume.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        completed_epoch=3,
        global_optimizer_step=11,
        data_order_state={"generator_state": torch.tensor([1, 2, 3])},
        data_order_exactness=DATA_ORDER_CALLER_ESTABLISHED,
        best_selection_ledger={"best_epoch": 2, "metric": "synthetic"},
        identity=identity,
        parent_resume_checkpoint_sha256=PARENT_SHA,
        continuation_index=4,
    )
    assert info.completed_epoch == 3
    assert info.global_optimizer_step == 11
    assert info.continuation_index == 4
    assert info.parent_resume_checkpoint_sha256 == PARENT_SHA
    assert info.data_order_exactness == DATA_ORDER_CALLER_ESTABLISHED
    assert info.size_bytes > 0
    assert info.sha256 == file_sha256(tmp_path / "latest_resume.pt")
    validate_latest_resume_checkpoint(
        tmp_path / "latest_resume.pt",
        expected_identity=identity,
        expected_parent_resume_checkpoint_sha256=PARENT_SHA,
    )

    restored_model = _model()
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.05, momentum=0.9)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    restored_scaler = MockScaler()
    state = restore_latest_resume_checkpoint(
        checkpoint_path=tmp_path / "latest_resume.pt",
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        expected_identity={"run_name": "NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST"},
        expected_parent_resume_checkpoint_sha256=PARENT_SHA,
    )
    _assert_state_dict_equal(restored_model.state_dict(), expected_model)
    _assert_nested_state_equal(restored_optimizer.state_dict(), expected_optimizer)
    _assert_nested_state_equal(restored_scheduler.state_dict(), expected_scheduler)
    assert restored_scaler.state_dict() == {"scale": 7.0}
    assert state["completed_epoch"] == 3
    assert state["global_optimizer_step"] == 11
    assert state["continuation_index"] == 4
    assert state["data_order_state_present"] is True
    assert state["data_order_exactness"] == DATA_ORDER_CALLER_ESTABLISHED
    assert state["best_selection_ledger"] == {"best_epoch": 2, "metric": "synthetic"}
    assert state["resume_point"] == RESUME_POINT
    assert "trusted internal research artifacts" in state["trusted_checkpoint_warning"]
    assert TRUSTED_CHECKPOINT_WARNING in state["trusted_checkpoint_warning"]
    assert random.random() == expected_rng_next[0]
    assert float(np.random.rand()) == expected_rng_next[1]
    assert torch.equal(torch.rand(1), expected_rng_next[2])
    if torch.cuda.is_available():
        assert state["rng"]["torch_cuda"] == "RESTORED"
    else:
        assert state["rng"]["torch_cuda"] == "NOT_CHECKED"

    _advance_once(restored_model, restored_optimizer)


def test_latest_resume_supports_explicit_none_scheduler_and_scaler(tmp_path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    save_latest_resume_checkpoint(
        checkpoint_path=tmp_path / "latest_resume.pt",
        model=model,
        optimizer=optimizer,
        completed_epoch=1,
        global_optimizer_step=2,
        scheduler=None,
        scaler=None,
    )
    payload, _info = load_latest_resume_checkpoint(tmp_path / "latest_resume.pt")
    assert payload["scheduler_state_dict"] is None
    assert payload["scaler_state_dict"] is None
    restore_model = _model()
    restore_optimizer = torch.optim.AdamW(restore_model.parameters(), lr=0.001)
    state = restore_latest_resume_checkpoint(
        checkpoint_path=tmp_path / "latest_resume.pt",
        model=restore_model,
        optimizer=restore_optimizer,
        restore_rng=False,
    )
    assert state["scheduler"] == "NONE"
    assert state["scaler"] == "NONE"
    assert state["rng"]["python"] == "NOT_CHECKED"


def test_strict_identity_type_comparison(tmp_path) -> None:
    _base_checkpoint(
        tmp_path,
        identity={
            "seed": 1,
            "enabled": True,
            "optional": None,
            "dataset_sha256": EXPECTED_HASH,
        },
    )
    validate_latest_resume_checkpoint(
        tmp_path / "latest_resume.pt",
        expected_identity={"dataset_sha256": EXPECTED_HASH},
    )
    with pytest.raises(ResumeCheckpointError, match="identity mismatch"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_identity={"seed": "1"},
        )
    with pytest.raises(ResumeCheckpointError, match="identity mismatch"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_identity={"enabled": 1},
        )
    with pytest.raises(ResumeCheckpointError, match="missing required identity"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_identity={"optional": "None"},
        )


def test_parent_sha_validation_and_self_parent_detection(tmp_path, monkeypatch) -> None:
    _base_checkpoint(tmp_path, parent_resume_checkpoint_sha256=None)
    validate_latest_resume_checkpoint(tmp_path / "latest_resume.pt")

    _base_checkpoint(tmp_path, parent_resume_checkpoint_sha256=PARENT_SHA)
    validate_latest_resume_checkpoint(
        tmp_path / "latest_resume.pt",
        expected_parent_resume_checkpoint_sha256=PARENT_SHA,
    )
    with pytest.raises(ResumeCheckpointError, match="parent_resume_checkpoint_sha256 mismatch"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_parent_resume_checkpoint_sha256=OTHER_PARENT_SHA,
        )

    with pytest.raises(ResumeCheckpointError, match="lowercase 64-hex"):
        _base_checkpoint(tmp_path, parent_resume_checkpoint_sha256="parent-sha")

    payload, info = load_latest_resume_checkpoint(tmp_path / "latest_resume.pt")
    payload["parent_resume_checkpoint_sha256"] = info.sha256
    torch.save(payload, tmp_path / "self_parent.pt")
    monkeypatch.setattr(resume_checkpoint, "file_sha256", lambda _path: info.sha256)
    with pytest.raises(ResumeCheckpointError, match="cannot equal current"):
        validate_latest_resume_checkpoint(tmp_path / "self_parent.pt")


def test_data_order_exactness_status(tmp_path) -> None:
    _base_checkpoint(tmp_path)
    payload, _info = load_latest_resume_checkpoint(tmp_path / "latest_resume.pt")
    assert payload["data_order_state_present"] is False
    assert payload["data_order_exactness"] == DATA_ORDER_NOT_ESTABLISHED

    _base_checkpoint(tmp_path, data_order_state={"epoch_order_token": "synthetic"})
    payload, _info = load_latest_resume_checkpoint(tmp_path / "latest_resume.pt")
    assert payload["data_order_state_present"] is True
    assert payload["data_order_exactness"] == DATA_ORDER_NOT_ESTABLISHED

    _base_checkpoint(
        tmp_path,
        data_order_state={"epoch_order_token": "synthetic"},
        data_order_exactness=DATA_ORDER_CALLER_ESTABLISHED,
    )
    payload, _info = load_latest_resume_checkpoint(tmp_path / "latest_resume.pt")
    assert payload["data_order_state_present"] is True
    assert payload["data_order_exactness"] == DATA_ORDER_CALLER_ESTABLISHED


def test_cuda_rng_device_count_fail_closed_without_gpu(monkeypatch) -> None:
    cuda_state = [torch.get_rng_state(), torch.get_rng_state()]
    rng_state = capture_rng_state(include_cuda=False)
    rng_state["torch_cuda"] = cuda_state
    rng_state["torch_cuda_device_count"] = 2

    called = {"set_all": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: called.__setitem__("set_all", states is cuda_state),
    )
    assert restore_rng_state(rng_state)["torch_cuda"] == "RESTORED"
    assert called["set_all"] is True

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ResumeCheckpointError, match="CUDA RNG device count mismatch"):
        restore_rng_state(rng_state)

    rng_state["torch_cuda_device_count"] = 3
    with pytest.raises(ResumeCheckpointError, match="CUDA RNG saved state count mismatch"):
        restore_rng_state(rng_state)
    rng_state["torch_cuda_device_count"] = 2

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ResumeCheckpointError, match="CUDA RNG continuity required"):
        restore_rng_state(rng_state, require_cuda_continuity=True)
    assert restore_rng_state(rng_state)["torch_cuda"] == "NOT_CHECKED_CUDA_UNAVAILABLE"

    rng_state["torch_cuda"] = None
    rng_state["torch_cuda_device_count"] = 0
    with pytest.raises(ResumeCheckpointError, match="checkpoint has no CUDA RNG state"):
        restore_rng_state(rng_state, require_cuda_continuity=True)


def test_failed_required_cuda_restore_does_not_mutate_cpu_rng(monkeypatch) -> None:
    random.seed(NON_SCIENTIFIC_SEED)
    np.random.seed(NON_SCIENTIFIC_SEED)
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    saved_state = capture_rng_state(include_cuda=False)
    saved_state["torch_cuda"] = [torch.get_rng_state()]
    saved_state["torch_cuda_device_count"] = 1

    random.seed(NON_SCIENTIFIC_SEED + 1)
    np.random.seed(NON_SCIENTIFIC_SEED + 1)
    torch.manual_seed(NON_SCIENTIFIC_SEED + 1)
    python_state_before = random.getstate()
    numpy_state_before = np.random.get_state()
    torch_cpu_state_before = torch.get_rng_state().clone()

    called = {"set_all": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _states: called.__setitem__("set_all", True),
    )
    with pytest.raises(ResumeCheckpointError, match="CUDA RNG continuity required"):
        restore_rng_state(saved_state, require_cuda_continuity=True)

    assert random.getstate() == python_state_before
    numpy_state_after = np.random.get_state()
    assert numpy_state_after[0] == numpy_state_before[0]
    assert np.array_equal(numpy_state_after[1], numpy_state_before[1])
    assert numpy_state_after[2:] == numpy_state_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_cpu_state_before)
    assert called["set_all"] is False


def test_portable_optional_payload_guard(tmp_path) -> None:
    _base_checkpoint(
        tmp_path,
        data_order_state={"state": [1, 2, torch.tensor([3]), np.array([4])]},
        best_selection_ledger={"history": [{"epoch": 1, "metric": 0.5}]},
        identity={"seed": NON_SCIENTIFIC_SEED, "hash": EXPECTED_HASH},
    )

    with pytest.raises(ResumeCheckpointError, match="best_selection_ledger contains a non-portable value"):
        _base_checkpoint(tmp_path, best_selection_ledger={"bad": CustomPayload()})

    with pytest.raises(ResumeCheckpointError, match="identity contains a non-portable value"):
        _base_checkpoint(tmp_path, identity={"bad": CustomPayload()})


def test_malformed_missing_field_and_identity_mismatch_fail_closed(tmp_path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        validate_latest_resume_checkpoint(bad)

    incomplete = tmp_path / "incomplete.pt"
    torch.save({"schema_version": "wrong"}, incomplete)
    with pytest.raises(ResumeCheckpointError, match="missing required"):
        validate_latest_resume_checkpoint(incomplete)

    _base_checkpoint(
        tmp_path,
        identity={"run_name": "expected", "seed": 9017},
        parent_resume_checkpoint_sha256=PARENT_SHA,
    )
    with pytest.raises(ResumeCheckpointError, match="identity mismatch"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_identity={"run_name": "wrong"},
        )
    with pytest.raises(ResumeCheckpointError, match="missing required identity"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_identity={"authority_sha": "required"},
        )
    with pytest.raises(ResumeCheckpointError, match="parent_resume_checkpoint_sha256 mismatch"):
        validate_latest_resume_checkpoint(
            tmp_path / "latest_resume.pt",
            expected_parent_resume_checkpoint_sha256=OTHER_PARENT_SHA,
        )


def test_atomic_temp_failure_does_not_corrupt_existing_latest_or_best(tmp_path, monkeypatch) -> None:
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    latest = tmp_path / "latest_resume.pt"
    best = tmp_path / "best_scientific_checkpoint.pt"
    best.write_bytes(b"BEST_SCIENTIFIC_CHECKPOINT fixture must remain untouched")
    save_latest_resume_checkpoint(
        checkpoint_path=latest,
        model=model,
        optimizer=optimizer,
        completed_epoch=1,
        global_optimizer_step=1,
    )
    latest_sha = file_sha256(latest)
    best_sha = file_sha256(best)

    def _raise_before_replace(_source, _target):
        raise OSError("synthetic interrupted replace")

    monkeypatch.setattr(os, "replace", _raise_before_replace)
    with pytest.raises(OSError, match="synthetic interrupted replace"):
        save_latest_resume_checkpoint(
            checkpoint_path=latest,
            model=model,
            optimizer=optimizer,
            completed_epoch=2,
            global_optimizer_step=2,
        )
    assert file_sha256(latest) == latest_sha
    assert file_sha256(best) == best_sha
    assert not list(tmp_path.glob(".latest_resume.pt.*.tmp"))
