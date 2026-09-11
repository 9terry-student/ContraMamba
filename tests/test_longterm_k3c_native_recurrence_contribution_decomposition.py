from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "longterm_k3c_native_recurrence_contribution_decomposition.py"
)
spec = importlib.util.spec_from_file_location("k3c_impl", MODULE_PATH)
assert spec is not None and spec.loader is not None
k3c = importlib.util.module_from_spec(spec)
sys.modules["k3c_impl"] = k3c
spec.loader.exec_module(k3c)


def test_frozen_prereg_authority():
    assert k3c.K3C_PREREG_AUTHORITY_COMMIT == (
        "b85272d88d0bb57db45fdc963d313714529e7975"
    )
    assert k3c.K3C_PREREG_SHA256 == (
        "0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436"
    )


def test_frozen_successor_authority():
    assert k3c.K3_SUCCESSOR_AUTHORITY_COMMIT == (
        "a8101ce9baf340f35c0c6b135f7312681805693c"
    )
    assert k3c.K3_SUCCESSOR_SHA256 == (
        "0040d3d3573faad4dfacb0b617b4626b5e1143a9aa946bd4d270fdf80c44523e"
    )


def test_frozen_population_identity():
    assert k3c.GLOBAL_TEMPLATE_START == 600
    assert k3c.GLOBAL_TEMPLATE_STOP == 900
    assert k3c.N_ITEMS == 300
    assert k3c.N_BLOCKS == 150
    assert k3c.GENERATED_SOURCE_CANONICAL_SHA256 == (
        "33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000"
    )
    assert k3c.CANDIDATE_POOL_CANONICAL_SHA256 == (
        "9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e"
    )
    assert k3c.RECIPROCAL_MAPPING_CANONICAL_SHA256 == (
        "4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc"
    )


def test_frozen_runtime_semantics():
    assert k3c.PRIMARY_LAYER == 23
    assert k3c.W == 8
    assert k3c.TRANSFORMERS_VERSION == "5.12.1"
    assert k3c.MAMBA_SOURCE_SHA256 == (
        "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
    )


def test_primary_order_exact():
    assert k3c.PRIMARY_TEST_ORDER == (
        "R_DOM", "R_CARRY",
        "D_DOM", "D_CARRY",
        "DISP_DOM", "DISP_CARRY",
        "P_DOM", "P_CARRY",
    )


def test_expected_direction_exact():
    assert k3c.EXPECTED_DIRECTION == {
        "R": 1, "D": 1, "DISP": -1, "P": -1
    }


def test_arithmetic_midpoint():
    a = torch.tensor([1.0, 3.0], dtype=torch.float32)
    b = torch.tensor([3.0, 5.0], dtype=torch.float32)
    assert torch.equal(
        k3c.arithmetic_midpoint(a, b),
        torch.tensor([2.0, 4.0]),
    )


def test_retained_contribution_is_g_times_state():
    g = torch.tensor([[[0.5]]], dtype=torch.float32)
    s = torch.tensor([[[4.0]]], dtype=torch.float32)
    assert torch.equal(
        k3c.retained_contribution(g, s),
        torch.tensor([[[2.0]]]),
    )


def test_structural_replay_known_terms():
    init = torch.zeros((1, 1, 1), dtype=torch.float32)
    g = {
        0: torch.full_like(init, 0.5),
        1: torch.full_like(init, 0.5),
    }
    w = {
        0: torch.ones_like(init),
        1: torch.ones_like(init),
    }
    out = k3c.structural_replay(init, g, w, 0, 1)
    assert torch.equal(out[0], torch.ones_like(init))
    assert torch.equal(out[1], torch.full_like(init, 1.5))


def _pair_terms():
    corr = {
        "G": {
            2: torch.tensor([[[0.5]]]),
            3: torch.tensor([[[0.5]]]),
            4: torch.tensor([[[0.5]]]),
        },
        "W": {
            2: torch.tensor([[[2.0]]]),
            3: torch.tensor([[[2.0]]]),
            4: torch.tensor([[[2.0]]]),
        },
        "post_state": {1: torch.tensor([[[0.0]]])},
    }
    ctrl = {
        "G": {
            2: torch.tensor([[[0.25]]]),
            3: torch.tensor([[[0.25]]]),
            4: torch.tensor([[[0.25]]]),
        },
        "W": {
            2: torch.tensor([[[0.0]]]),
            3: torch.tensor([[[0.0]]]),
            4: torch.tensor([[[0.0]]]),
        },
        "post_state": {1: torch.tensor([[[0.0]]])},
    }
    return corr, ctrl


def test_base_semantics():
    corr, ctrl = _pair_terms()
    c, n = k3c.replay_pair_contribution(corr, ctrl, 2, 4, "BASE")
    assert torch.equal(c[2], torch.tensor([[[2.0]]]))
    assert torch.equal(n[2], torch.tensor([[[0.0]]]))
    assert torch.equal(c[3], torch.tensor([[[3.0]]]))


def test_w_eq_equalizes_write_each_step():
    corr, ctrl = _pair_terms()
    c, n = k3c.replay_pair_contribution(corr, ctrl, 2, 4, "W_EQ")
    assert torch.equal(c[2], torch.tensor([[[1.0]]]))
    assert torch.equal(n[2], torch.tensor([[[1.0]]]))
    # Natural G remains branch-specific, so later states can diverge.
    assert not torch.equal(c[3], n[3])


def test_h_eq_equalizes_dynamic_retained_contribution():
    corr, ctrl = _pair_terms()
    c, n = k3c.replay_pair_contribution(corr, ctrl, 2, 4, "H_EQ")
    # At every step, the state difference is exactly the natural W difference.
    for t in (2, 3, 4):
        assert torch.equal(c[t] - n[t], torch.tensor([[[2.0]]]))


def test_w_seed_h_carry_seeds_only_at_d_and_carries_later():
    corr, ctrl = _pair_terms()
    c, n = k3c.replay_pair_contribution(
        corr, ctrl, 2, 4, "W_SEED_H_CARRY"
    )
    assert torch.equal(c[2], torch.tensor([[[2.0]]]))
    assert torch.equal(n[2], torch.tensor([[[0.0]]]))
    # From t=3 onward W is midpoint-equalized, yet divergence persists via H.
    assert torch.equal(c[3], torch.tensor([[[2.0]]]))
    assert torch.equal(n[3], torch.tensor([[[1.0]]]))
    assert not torch.equal(c[4], n[4])


def test_wh_eq_collapses_pair_exactly():
    corr, ctrl = _pair_terms()
    c, n = k3c.replay_pair_contribution(corr, ctrl, 2, 4, "WH_EQ")
    for t in (2, 3, 4):
        assert torch.equal(c[t], n[t])


def test_pair_initial_state_must_match():
    corr, ctrl = _pair_terms()
    ctrl["post_state"][1] = torch.tensor([[[1.0]]])
    with pytest.raises(k3c.ContractError):
        k3c.replay_pair_contribution(corr, ctrl, 2, 4, "WH_EQ")


def test_invalid_condition_rejected():
    corr, ctrl = _pair_terms()
    with pytest.raises(k3c.ContractError):
        k3c.replay_pair_contribution(corr, ctrl, 2, 4, "G_EQ")


def test_known_term_source_carry_control():
    result = k3c.synthetic_known_term_source_carry_control()
    assert result["known_term_source_carry_replay"] == "PASS_EXACT"
    assert result["known_term_h_eq"] == "PASS_EXACT"
    assert result["known_term_wh_eq_collapse"] == "PASS_EXACT"


def test_aligned_signal():
    assert k3c.aligned_signal("R", 2.0) == 2.0
    assert k3c.aligned_signal("D", -2.0) == -2.0
    assert k3c.aligned_signal("DISP", -2.0) == 2.0
    assert k3c.aligned_signal("P", 3.0) == -3.0


def test_mechanism_values_dom_and_carry():
    out = k3c.mechanism_values(
        z_base=10.0,
        z_w_eq=2.0,
        z_h_eq=7.0,
        z_carry=5.0,
    )
    assert out["ATT_W"] == 8.0
    assert out["ATT_H"] == 3.0
    assert out["DOM"] == 5.0
    assert out["CARRY"] == 5.0


def test_exact_sign_p():
    assert k3c.exact_two_sided_sign_p(10, 10) == 1.0
    assert k3c.exact_two_sided_sign_p(20, 0) < 1e-5


def test_floor_failure_raw_p_one():
    out = k3c.summarize_test([1.0] * 20 + [-1.0] * 10)
    assert out["promotion_floor_pass"] is False
    assert out["raw_p"] == 1.0


def test_holm_m4():
    raw = {name: 0.01 for name in k3c.BASE_ORDER}
    out = k3c.holm_adjust(raw, k3c.BASE_ORDER)
    assert out["R"]["holm_adjusted_p"] == pytest.approx(0.04)


def test_holm_m8():
    raw = {name: 0.01 for name in k3c.PRIMARY_TEST_ORDER}
    out = k3c.holm_adjust(raw, k3c.PRIMARY_TEST_ORDER)
    assert out["R_DOM"]["holm_adjusted_p"] == pytest.approx(0.08)


def _strong_positive():
    return [1.0] * 140 + [-1.0] * 10


def _strong_negative():
    return [-1.0] * 140 + [1.0] * 10


def test_base_replication_pass_requires_expected_metric_directions():
    values = {
        "R": _strong_positive(),
        "D": _strong_positive(),
        "DISP": _strong_negative(),
        "P": _strong_negative(),
    }
    out = k3c.base_replication_statistics(values)
    assert out["base_replication_verdict"] == "PASS"
    assert out["direction_matched_endpoints"] == list(k3c.BASE_ORDER)


def test_base_replication_contradiction():
    values = {
        "R": _strong_positive(),
        "D": _strong_positive(),
        "DISP": _strong_positive(),
        "P": _strong_negative(),
    }
    out = k3c.base_replication_statistics(values)
    assert out["base_replication_verdict"] == "CONTRADICTED"
    assert "DISP" in out["directional_contradiction_endpoints"]


def test_mechanism_full_support_requires_eight_of_eight():
    values = {
        name: _strong_positive()
        for name in k3c.PRIMARY_TEST_ORDER
    }
    out = k3c.mechanism_statistics(values)
    assert out["full_support"] is True
    assert out["mechanism_verdict"] == k3c.SUCCESS_VERDICT


def test_mechanism_seven_of_eight_not_established():
    values = {
        name: _strong_positive()
        for name in k3c.PRIMARY_TEST_ORDER
    }
    values["P_CARRY"] = [1.0] * 75 + [-1.0] * 75
    out = k3c.mechanism_statistics(values)
    assert out["full_support"] is False
    assert out["mechanism_verdict"] == k3c.NOT_ESTABLISHED_VERDICT


def test_mechanism_negative_is_contradiction():
    values = {
        name: _strong_positive()
        for name in k3c.PRIMARY_TEST_ORDER
    }
    values["D_CARRY"] = _strong_negative()
    out = k3c.mechanism_statistics(values)
    assert out["mechanism_verdict"] == k3c.CONTRADICTION_VERDICT
    assert "D_CARRY" in out["directional_contradiction_tests"]


def test_base_failure_blocks_mechanism_promotion():
    base = {
        "R": [1.0] * 75 + [-1.0] * 75,
        "D": _strong_positive(),
        "DISP": _strong_negative(),
        "P": _strong_negative(),
    }
    mechanism = {
        name: _strong_positive()
        for name in k3c.PRIMARY_TEST_ORDER
    }
    out = k3c.k3c_scientific_statistics(base, mechanism)
    assert out["mechanism"]["full_support"] is True
    assert out["mechanism_authorized_for_promotion"] is False
    assert out["scientific_verdict"] == k3c.BASE_FAILURE_VERDICT


def test_base_pass_allows_mechanism_success():
    base = {
        "R": _strong_positive(),
        "D": _strong_positive(),
        "DISP": _strong_negative(),
        "P": _strong_negative(),
    }
    mechanism = {
        name: _strong_positive()
        for name in k3c.PRIMARY_TEST_ORDER
    }
    out = k3c.k3c_scientific_statistics(base, mechanism)
    assert out["mechanism_authorized_for_promotion"] is True
    assert out["scientific_verdict"] == k3c.SUCCESS_VERDICT


def test_first_divergence_after_prefix():
    assert k3c.first_divergence_after_prefix(
        [1, 2, 3, 4, 9],
        [1, 2, 3, 4, 8],
        4,
    ) == 4


def test_cli_requires_replay_preflight():
    parser = k3c.parser()
    args = parser.parse_args([
        "--seed180-handoff", "x.zip",
        "--hf-revision", k3c.HF_REVISION,
    ])
    assert args.replay_preflight is False


def test_scientific_execution_not_reachable_by_parser():
    parser = k3c.parser()
    actions = {a.dest for a in parser._actions}
    assert "output_dir" not in actions
    assert "scientific_execution" not in actions
    assert "replay_preflight" in actions
