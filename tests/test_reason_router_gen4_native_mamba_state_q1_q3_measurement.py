from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


PATH = Path("scripts/reason_router_gen4_native_mamba_state_q1_q3_measurement.py")


def load():
    spec = importlib.util.spec_from_file_location("q_measurement", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scientific_state(value: float = 0.0) -> torch.Tensor:
    return torch.full((1, 1536, 16), float(value), dtype=torch.float32, device="cpu")


def test_authority_and_primary_identity_binding():
    m = load()
    assert m.IMPLEMENTATION_AUTHORITY_COMMIT == "c395e634448a3b37c30053d89abef37c5a269afe"
    assert m.PRIMARY_MEASUREMENT_SHA256 == "7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c"
    m._primary_identity()


def test_primary_identity_accepts_crlf_checkout_equivalent(monkeypatch, tmp_path):
    m = load()
    canonical = m.PRIMARY_MEASUREMENT_PATH.read_bytes().replace(b"\r\n", b"\n")
    candidate = tmp_path / "primary.py"
    candidate.write_bytes(canonical.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(m, "PRIMARY_MEASUREMENT_PATH", candidate)
    m._primary_identity()


def test_exact_registration_and_rejections():
    m = load()
    m.validate_secondary_layer_registration({101: {"layer_index": 5}, 202: {"layer_index": 17}})
    bad_cases = (
        {101: {"layer_index": 11}, 202: {"layer_index": 17}},
        {101: {"layer_index": 5}},
        {101: {"layer_index": 5}, 202: {"layer_index": 5}},
        {101: {"layer_index": 5}, 202: {"layer_index": 18}},
        {True: {"layer_index": 5}, 202: {"layer_index": 17}},
    )
    for bad in bad_cases:
        with pytest.raises(m.ContractError):
            m.validate_secondary_layer_registration(bad)


def test_observer_is_default_disabled():
    m = load()
    assert m.observer_default_enabled_value() is False
    assert inspect.signature(m.DualLayerNativeStateObserver.__init__).parameters["enabled"].default is False


def test_coordinates_are_exact_ordered_dual_layer():
    m = load()
    snapshots = {
        (1, layer, token): scientific_state(layer + token)
        for layer in (5, 17)
        for token in range(2)
    }
    result = m.validate_capture_coordinates(snapshots, 2)
    assert tuple(result) == (5, 17)
    assert len(result[5]) == 2 and len(result[17]) == 2

    missing = dict(snapshots)
    del missing[(1, 5, 1)]
    with pytest.raises(m.ContractError):
        m.validate_capture_coordinates(missing, 2)

    reordered = {key: snapshots[key] for key in reversed(list(snapshots))}
    with pytest.raises(m.ContractError):
        m.validate_capture_coordinates(reordered, 2)


def test_coordinate_state_shape_dtype_and_finiteness_fail_closed():
    m = load()
    wrong_shape = {
        (1, layer, 0): torch.zeros((1, 10, 10), dtype=torch.float32)
        for layer in (5, 17)
    }
    with pytest.raises(m.ContractError):
        m.validate_capture_coordinates(wrong_shape, 1)

    wrong_dtype = {
        (1, layer, 0): torch.zeros((1, 1536, 16), dtype=torch.float64)
        for layer in (5, 17)
    }
    with pytest.raises(m.ContractError):
        m.validate_capture_coordinates(wrong_dtype, 1)

    nonfinite = {(1, layer, 0): scientific_state() for layer in (5, 17)}
    nonfinite[(1, 17, 0)][0, 0, 0] = float("nan")
    with pytest.raises(m.ContractError):
        m.validate_capture_coordinates(nonfinite, 1)


def test_same_layer_post4_and_blockers():
    m = load()
    states = [scientific_state(i) for i in range(8)]
    values = m.endpoint_from_same_layer(states, 1, 5)
    assert set(values) == {"POST4_SPEED", "POST4_TURNING", "POST4_PATH_EFFICIENCY"}
    assert values["POST4_SPEED"] > 0
    assert values["POST4_TURNING"] == pytest.approx(0.0, abs=1e-6)
    assert values["POST4_PATH_EFFICIENCY"] == pytest.approx(1.0)
    with pytest.raises(m.ContractError):
        m.endpoint_from_same_layer(states, 1, 11)
    with pytest.raises(m.ContractError):
        m.endpoint_from_same_layer([scientific_state(0.0) for _ in range(8)], 1, 5)
