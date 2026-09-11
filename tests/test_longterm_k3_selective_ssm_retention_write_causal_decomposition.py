from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "longterm_k3_selective_ssm_retention_write_causal_decomposition.py"
)
spec = importlib.util.spec_from_file_location("k3_impl", MODULE_PATH)
assert spec is not None and spec.loader is not None
k3 = importlib.util.module_from_spec(spec)
import sys
sys.modules["k3_impl"] = k3
spec.loader.exec_module(k3)


def test_frozen_authority_constants():
    assert k3.K3_PREREG_AUTHORITY_COMMIT == "20032bb53d77416bb7eb25411eb4f77b008648b4"
    assert k3.K3_PREREG_SHA256 == "7561f4188921b645eba3d7108bcd007b7c223389b5e1abc512cc8d61af976c84"


def test_frozen_k2r_dependencies():
    assert k3.K2R_RUNNER_SHA256 == "557d537e3dde30fdfd1f3e03fe9a2e7d019499e8e53c76242ff5eee9b8c87eeb"
    assert k3.K2R_CANDIDATE_SHA256 == "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4"
    assert k3.K2R_BLOCK_SHA256 == "e4982e8a57e15863227d17f080e7a7a699fb14100387fc36c7c47ab68962c8de"


def test_frozen_runtime_semantics():
    assert k3.TRANSFORMERS_VERSION == "5.12.1"
    assert k3.MAMBA_SOURCE_SHA256 == "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
    assert k3.PRIMARY_LAYER == 23
    assert k3.W == 8


def test_primary_component_specialization():
    assert k3.DOMINANT_COMPONENT == {"R": "W", "D": "W", "DISP": "G", "P": "G"}
    assert k3.EXPECTED_DIRECTION == {"R": 1, "D": 1, "DISP": -1, "P": -1}


def test_primary_test_order_exact():
    assert k3.PRIMARY_TEST_ORDER == (
        "R_ATT", "R_SEL", "D_ATT", "D_SEL",
        "DISP_ATT", "DISP_SEL", "P_ATT", "P_SEL",
    )


def _source(update="ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]"):
    return f"""
import torch
class MambaMixer:
    def slow_forward(self):
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        for i in range(seq_len):
            {update}
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
"""


def test_ast_component_binding_accepts_exact_structure():
    result = k3.analyze_component_source(_source().encode())
    assert result["recurrence_update_line"] + 1 == result["post_update_line"]
    assert result["discrete_A_line"] < result["recurrence_update_line"]
    assert result["deltaB_u_line"] < result["recurrence_update_line"]


def test_ast_component_binding_rejects_wrong_update():
    with pytest.raises(k3.ContractError):
        k3.analyze_component_source(
            _source("ssm_state = discrete_A[:, :, i, :] + ssm_state + deltaB_u[:, :, i, :]").encode()
        )


def test_first_divergence():
    corr = [1, 2, 3, 4, 9, 6]
    ctrl = [1, 2, 3, 4, 8, 6]
    assert k3.first_divergence(corr, ctrl, 4, 5) == 4


def test_arithmetic_midpoint_exact():
    a = torch.tensor([1.0, 3.0], dtype=torch.float32)
    b = torch.tensor([3.0, 5.0], dtype=torch.float32)
    assert torch.equal(k3.arithmetic_midpoint(a, b), torch.tensor([2.0, 4.0]))


def test_structural_replay_known_terms():
    init = torch.zeros((1, 1, 1), dtype=torch.float32)
    G = {0: torch.full_like(init, 0.5), 1: torch.full_like(init, 0.5)}
    W = {0: torch.ones_like(init), 1: torch.ones_like(init)}
    out = k3.structural_replay(init, G, W, 0, 1)
    assert torch.equal(out[0], torch.ones_like(init))
    assert torch.equal(out[1], torch.full_like(init, 1.5))


def _pair_terms():
    shape = (1, 1, 1)
    corr = {
        "G": {2: torch.tensor([[[0.2]]]), 3: torch.tensor([[[0.3]]])},
        "W": {2: torch.tensor([[[1.0]]]), 3: torch.tensor([[[2.0]]])},
        "post_state": {1: torch.tensor([[[4.0]]])},
    }
    ctrl = {
        "G": {2: torch.tensor([[[0.6]]]), 3: torch.tensor([[[0.7]]])},
        "W": {2: torch.tensor([[[3.0]]]), 3: torch.tensor([[[4.0]]])},
        "post_state": {1: torch.tensor([[[4.0]]])},
    }
    return corr, ctrl


def test_w_equalization_semantics():
    corr, ctrl = _pair_terms()
    cg, cw, ng, nw = k3.equalized_terms(
        corr["G"], corr["W"], ctrl["G"], ctrl["W"], 2, 3, "W_EQ"
    )
    assert torch.equal(cg[2], corr["G"][2])
    assert torch.equal(ng[2], ctrl["G"][2])
    assert torch.equal(cw[2], nw[2])
    assert torch.equal(cw[2], torch.tensor([[[2.0]]]))


def test_g_equalization_semantics():
    corr, ctrl = _pair_terms()
    cg, cw, ng, nw = k3.equalized_terms(
        corr["G"], corr["W"], ctrl["G"], ctrl["W"], 2, 3, "G_EQ"
    )
    assert torch.equal(cg[2], ng[2])
    assert torch.equal(cg[2], torch.tensor([[[0.4]]]))
    assert torch.equal(cw[2], corr["W"][2])
    assert torch.equal(nw[2], ctrl["W"][2])


def test_gw_equalization_collapses_pair():
    corr, ctrl = _pair_terms()
    a, b = k3.replay_pair(corr, ctrl, 2, 3, "GW_EQ")
    assert torch.equal(a[2], b[2])
    assert torch.equal(a[3], b[3])


def test_base_replay_preserves_branch_specific_terms():
    corr, ctrl = _pair_terms()
    a, b = k3.replay_pair(corr, ctrl, 2, 3, "BASE")
    expected_a2 = corr["G"][2] * corr["post_state"][1] + corr["W"][2]
    expected_b2 = ctrl["G"][2] * ctrl["post_state"][1] + ctrl["W"][2]
    assert torch.equal(a[2], expected_a2)
    assert torch.equal(b[2], expected_b2)


def test_pair_initial_state_must_match():
    corr, ctrl = _pair_terms()
    ctrl["post_state"][1] = torch.tensor([[[5.0]]])
    with pytest.raises(k3.ContractError):
        k3.replay_pair(corr, ctrl, 2, 3, "GW_EQ")


def test_aligned_signal():
    assert k3.aligned_signal("R", 2.0) == 2.0
    assert k3.aligned_signal("P", -2.0) == 2.0
    assert k3.aligned_signal("DISP", 3.0) == -3.0


def test_local_metric_dominant_attenuation_is_w():
    out = k3.attenuation_and_selectivity("R", 10.0, 4.0, 8.0)
    assert out["ATT_W"] == 6.0
    assert out["ATT_G"] == 2.0
    assert out["ATT_DOM"] == 6.0
    assert out["SEL"] == 4.0


def test_net_metric_dominant_attenuation_is_g():
    out = k3.attenuation_and_selectivity("P", 10.0, 7.0, 2.0)
    assert out["ATT_W"] == 3.0
    assert out["ATT_G"] == 8.0
    assert out["ATT_DOM"] == 8.0
    assert out["SEL"] == 5.0


def test_exact_sign_p_symmetric():
    assert k3.exact_two_sided_sign_p(10, 10) == 1.0
    assert k3.exact_two_sided_sign_p(20, 0) < 1e-5


def test_support_floor_fail_sets_raw_p_one():
    summary = k3.summarize_test([1.0] * 20 + [-1.0] * 10)
    assert summary["promotion_floor_pass"] is False
    assert summary["raw_p"] == 1.0


def test_holm_eight_tie_order_and_monotonicity():
    raw = {name: 0.01 for name in k3.PRIMARY_TEST_ORDER}
    out = k3.holm_adjust_eight(raw)
    vals = [out[name]["holm_adjusted_p"] for name in k3.PRIMARY_TEST_ORDER]
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(0.08)


def _all_values(pos=True):
    if pos:
        return [1.0] * 140 + [-1.0] * 10
    return [-1.0] * 140 + [1.0] * 10


def test_full_success_requires_eight_of_eight():
    values = {name: _all_values(True) for name in k3.PRIMARY_TEST_ORDER}
    result = k3.k3_primary_statistics(values)
    assert result["full_support"] is True
    assert result["scientific_verdict"] == k3.SUCCESS_VERDICT
    assert result["direction_matched_tests"] == list(k3.PRIMARY_TEST_ORDER)


def test_seven_of_eight_is_not_full_support():
    values = {name: _all_values(True) for name in k3.PRIMARY_TEST_ORDER}
    values["P_SEL"] = [1.0] * 75 + [-1.0] * 75
    result = k3.k3_primary_statistics(values)
    assert result["full_support"] is False
    assert result["scientific_verdict"] == k3.NOT_ESTABLISHED_VERDICT


def test_significant_negative_is_contradiction():
    values = {name: _all_values(True) for name in k3.PRIMARY_TEST_ORDER}
    values["D_SEL"] = _all_values(False)
    result = k3.k3_primary_statistics(values)
    assert result["full_support"] is False
    assert result["scientific_verdict"] == k3.CONTRADICTION_VERDICT
    assert "D_SEL" in result["directional_contradiction_tests"]


def test_cli_requires_replay_preflight_flag():
    parser = k3.parser()
    args = parser.parse_args([
        "--seed180-handoff", "x.zip",
        "--hf-revision", k3.HF_REVISION,
    ])
    assert args.replay_preflight is False


def test_scientific_execution_not_reachable_by_parser():
    parser = k3.parser()
    actions = {a.dest for a in parser._actions}
    assert "output_dir" not in actions
    assert "scientific_execution" not in actions
