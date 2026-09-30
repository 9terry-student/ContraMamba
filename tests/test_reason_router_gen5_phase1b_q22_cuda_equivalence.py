from __future__ import annotations

import math

import torch

from scripts import reason_router_gen5_phase1b_q22_cuda_equivalence as impl


def synthetic_bases():
    r22 = torch.zeros((impl.STATE_WIDTH, 2), dtype=torch.float64)
    c22 = torch.zeros((impl.STATE_WIDTH, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    return r22, c22


def fake_update(state, u, delta, a_matrix, b, c, d, gate, bias, dt_softplus=True):
    del delta, a_matrix, c, d, gate, bias, dt_softplus
    state.add_(u.unsqueeze(-1) * b.unsqueeze(1))
    return torch.zeros_like(u)


def fake_scan(u, delta, a_matrix, b, c, d, gate, bias, delta_softplus=True, return_last_state=True):
    del delta_softplus, return_last_state
    state = torch.zeros((u.shape[0], u.shape[1], b.shape[1]), dtype=u.dtype, device=u.device)
    for token in range(u.shape[-1]):
        fake_update(state, u[..., token], delta[..., token], a_matrix, b[..., token], c[..., token], d, gate[..., token], bias)
    return torch.zeros_like(u), state


def captured_fixture():
    seq = 8
    u = torch.zeros((1, 1536, seq), dtype=torch.float32)
    for token in range(seq):
        u[0, 0, token] = 1.0 + token
        u[0, 1, token] = 2.0 + token
        u[0, 2, token] = 3.0 + token
        u[0, 3, token] = 4.0 + token
    delta = torch.ones_like(u)
    a = torch.zeros((1536, 16), dtype=torch.float32)
    b = torch.ones((1, 16, seq), dtype=torch.float32)
    c = torch.ones((1, 16, seq), dtype=torch.float32)
    d = torch.zeros((1536,), dtype=torch.float32)
    gate = torch.ones_like(u)
    bias = torch.zeros((1536,), dtype=torch.float32)
    return (u, delta, a, b, c, d, gate, bias)


def test_frozen_budget_and_tolerances():
    assert impl.PAIR_ID == "xg1_fact_7801"
    assert impl.FORWARDS_PER_BACKEND == 120
    assert impl.TOTAL_EQUIVALENCE_MODEL_FORWARDS == 240
    assert impl.WRITE_ATOL == 1e-4
    assert impl.STATE_ATOL == 1e-4
    assert impl.PE22_ATOL == 1e-4
    assert impl.F22_ATOL == 2e-4
    assert impl.J22_ATOL == 0.008


def test_j2_bound():
    ref = 1.25
    assert impl.j2_error_bound(ref) == 2.0 * abs(ref) * 0.008 + 0.008 ** 2


def test_replay_native_r22_c22_semantics():
    r22, c22 = synthetic_bases()
    captured = captured_fixture()
    anchor = 2
    native = impl.replay_q22_window(captured, anchor=anchor, condition="native22", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)
    r = impl.replay_q22_window(captured, anchor=anchor, condition="r22_neutralized", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)
    cctrl = impl.replay_q22_window(captured, anchor=anchor, condition="c22_coefficient_control", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)
    assert torch.allclose(r["actual_write"].reshape(-1)[:2], torch.zeros(2))
    assert torch.allclose(cctrl["actual_write"].reshape(-1)[:2], native["native_write"].reshape(-1)[:2])
    assert math.isclose(r["plan"]["r_correction_l2"], cctrl["plan"]["c_correction_l2"], abs_tol=1e-12)
    assert set(native["post_states"]) == set(range(anchor, anchor + 5))


def test_tensor_equivalence_limit():
    ref = torch.tensor([1.0, 2.0], dtype=torch.float32)
    acc = torch.tensor([1.0, 2.0001], dtype=torch.float32)
    result = impl.tensor_equivalence(ref, acc, atol=1e-4, rtol=1e-4, label="TEST")
    assert result["max_abs_diff"] <= result["limit"]


def test_q22_replay_path_is_finite():
    r22, c22 = synthetic_bases()
    result = impl.replay_q22_window(captured_fixture(), anchor=2, condition="native22", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)
    assert math.isfinite(result["pe22"])
    assert 0.0 < result["pe22"] <= 1.0


def test_frozen_write_rule_identity():
    assert impl.necessity.R22_SHA256 == "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
    assert impl.necessity.C22_SHA256 == "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"
    assert impl.CONDITIONS == impl.necessity.CONDITIONS
    assert impl.DIRECTIONS == impl.necessity.DIRECTIONS
