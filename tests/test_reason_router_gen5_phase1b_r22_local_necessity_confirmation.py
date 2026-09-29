from __future__ import annotations

import math

import torch

from scripts import reason_router_gen5_phase1b_r22_local_necessity_confirmation as impl


def _synthetic_bases():
    r22 = torch.zeros((impl.STATE_WIDTH, 2), dtype=torch.float64)
    c22 = torch.zeros((impl.STATE_WIDTH, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    return r22, c22


def _synthetic_write():
    w = torch.zeros(impl.STATE_SHAPE, dtype=torch.float32)
    w.reshape(-1)[:4] = torch.tensor([3.0, 4.0, 5.0, 6.0])
    return w


def test_coefficient_transfer_exact_semantics():
    r22, c22 = _synthetic_bases()
    w = _synthetic_write()
    native = impl.intervention_vectors(w, r22, c22, "native22")
    r = impl.intervention_vectors(w, r22, c22, "r22_neutralized")
    c = impl.intervention_vectors(w, r22, c22, "c22_coefficient_control")

    assert torch.allclose(native["modified"], native["native"])
    assert torch.allclose(r["a"], torch.tensor([3.0, 4.0], dtype=torch.float64))
    assert torch.allclose(r["modified"][:2], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(c["modified"][:2], torch.tensor([3.0, 4.0], dtype=torch.float64))
    assert torch.allclose(c["modified"][2:4], torch.tensor([2.0, 2.0], dtype=torch.float64))
    assert math.isclose(r["r_correction_l2"], 5.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(c["c_correction_l2"], 5.0, rel_tol=0.0, abs_tol=1e-12)
    assert r["norm_equality_residual"] <= r["norm_equality_tolerance"]


def test_endpoint_identity():
    row = impl.endpoint(1.2, 0.7, 0.9)
    assert row["A_R"] == 0.5
    assert math.isclose(row["A_C"], 0.3, abs_tol=1e-15)
    assert math.isclose(row["D_NEC22"], 0.2, abs_tol=1e-15)
    assert math.isclose(row["A_R"] - row["A_C"], row["D_NEC22"], abs_tol=1e-15)


def test_student_t_and_single_confirmatory_decision():
    values = [0.1 + 0.001 * i for i in range(300)]
    test = impl.one_sided_one_sample_student_t(values)
    assert test["degrees_of_freedom"] == 299
    assert test["p_one_sided_greater"] < 0.05

    items = [
        {"Q0": 1.0, "A_R": 0.2, "D_NEC22": value, "row_dropped": False}
        for value in values
    ]
    decision = impl.confirmatory_decision(items)
    assert decision["confirmatory_p_value_count"] == 1
    assert decision["label"] == impl.LABEL_SUPPORTED


def test_decision_fails_when_gate_mean_not_positive():
    values = [0.1 + 0.001 * i for i in range(300)]
    items = [
        {"Q0": -0.1, "A_R": 0.2, "D_NEC22": value, "row_dropped": False}
        for value in values
    ]
    decision = impl.confirmatory_decision(items)
    assert decision["confirmatory_p_value_count"] == 1
    assert decision["label"] == impl.LABEL_NOT_ESTABLISHED


def test_q22_path_efficiency_reuses_layer22_trajectory_function():
    anchor = 2
    states = {}
    for offset in range(5):
        s = torch.zeros(impl.STATE_SHAPE, dtype=torch.float32)
        s.reshape(-1)[0] = float(offset)
        s.reshape(-1)[1] = float(offset * offset)
        states[anchor + offset] = s
    observed = impl.q22_path_efficiency(states, anchor)

    first = states[anchor].reshape(-1).numpy().copy()
    full = [first.copy() for _ in range(anchor + 5)]
    for token in range(anchor, anchor + 5):
        full[token] = states[token].reshape(-1).numpy().copy()
    expected = impl.core.post4_path_efficiency(full, anchor)
    assert observed == expected


def test_synthetic_trace_modifies_only_target_write_before_recurrence():
    r22, c22 = _synthetic_bases()

    class Mixer:
        pass

    class Frame:
        pass

    mixer = Mixer()
    code = object()
    anchor = 2
    collector = impl.Layer22Q22Collector(
        code=code,
        update_line=10,
        readout_line=11,
        mixer22=mixer,
        anchor=anchor,
        condition="r22_neutralized",
        r22=r22,
        c22=c22,
    )
    collector._pending = {}
    collector.post_states = {}
    collector.target_audit = None

    untouched_hashes = {}
    for token in range(anchor, anchor + 5):
        frame = Frame()
        frame.f_code = code
        frame.f_lineno = 10
        s_prev = torch.zeros(impl.STATE_SHAPE, dtype=torch.float32)
        discrete_a = torch.zeros((1, 1536, anchor + 5, 16), dtype=torch.float32)
        delta_b_u = torch.zeros((1, 1536, anchor + 5, 16), dtype=torch.float32)
        discrete_a[:, :, token, :].fill_(0.5)
        offset = float(token - anchor)
        delta_b_u[0, 0, token, :4] = torch.tensor(
            [
                3.0 + offset,
                4.0 + 2.0 * offset,
                5.0 + 3.0 * offset,
                6.0 + 4.0 * offset,
            ]
        )
        before = delta_b_u.clone()
        frame.f_locals = {
            "self": mixer,
            "i": token,
            "ssm_state": s_prev,
            "discrete_A": discrete_a,
            "deltaB_u": delta_b_u,
        }

        collector._trace(frame, "line", None)

        target = anchor + impl.core.TARGET_OFFSET
        if token == target:
            assert torch.allclose(
                delta_b_u[0, 0, token, :2], torch.zeros(2)
            )
            before[:, :, token, :].copy_(delta_b_u[:, :, token, :])
            assert torch.equal(before, delta_b_u)
        else:
            assert torch.equal(before, delta_b_u)

        actual = delta_b_u[:, :, token, :].clone()
        s_post = 0.5 * s_prev + actual
        frame.f_lineno = 11
        frame.f_locals["ssm_state"] = s_post
        collector._trace(frame, "line", None)

    assert set(collector.post_states or {}) == set(range(anchor, anchor + 5))
    assert collector.target_audit is not None
    assert collector.target_audit["condition"] == "r22_neutralized"
    assert collector.target_audit["target_recurrence_relative_residual"] <= impl.RECURRENCE_REL_TOL


def test_forward_budget_and_population_are_frozen():
    assert impl.PAIR_COUNT == 300
    assert impl.PAIR_FIRST == 8101
    assert impl.PAIR_LAST == 8400
    assert impl.FORWARDS_PER_CONDITION == 40
    assert impl.FORWARDS_PER_PAIR == 120
    assert impl.FULL_FORWARD_BUDGET == 36000
    assert impl.CONDITIONS == (
        "native22",
        "r22_neutralized",
        "c22_coefficient_control",
    )
