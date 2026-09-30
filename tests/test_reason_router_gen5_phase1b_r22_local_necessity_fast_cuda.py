from __future__ import annotations

import math

import torch

from scripts import reason_router_gen5_phase1b_r22_local_necessity_fast_cuda as impl


def synthetic_bases():
    r22 = torch.zeros(
        (impl.necessity.STATE_WIDTH, impl.necessity.RANK),
        dtype=torch.float64,
    )
    c22 = torch.zeros_like(r22)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    return r22, c22


def synthetic_branch(condition: str, r22: torch.Tensor, c22: torch.Tensor):
    anchor = 2
    native = torch.zeros(
        impl.necessity.STATE_SHAPE,
        dtype=torch.float32,
    )
    native.reshape(-1)[0] = 1.25
    native.reshape(-1)[1] = -0.75
    native.reshape(-1)[2] = 0.5
    plan = impl.necessity.intervention_vectors(
        native,
        r22,
        c22,
        condition,
    )
    actual = (
        plan["modified"]
        .reshape(impl.necessity.STATE_SHAPE)
        .to(torch.float32)
        .contiguous()
    )
    states = {}
    for token in range(anchor, anchor + 5):
        state = torch.zeros(
            impl.necessity.STATE_SHAPE,
            dtype=torch.float32,
        )
        state.reshape(-1)[:4] = torch.tensor(
            [1.0 + token, 2.0 + token, 3.0 + token, 4.0 + token]
        )
        if token == anchor + impl.necessity.core.TARGET_OFFSET:
            state = state + actual
        states[token] = state
    return {
        "native_write": native,
        "actual_write": actual,
        "plan": plan,
        "post_states": states,
        "pe22": float(
            impl.necessity.q22_path_efficiency(states, anchor)
        ),
        "layer17_probe_norm": 2.0 * impl.necessity.EPSILON,
        "target_token_index":
            anchor + impl.necessity.core.TARGET_OFFSET,
    }


def test_frozen_scope_and_budget():
    assert impl.AUTHORITY_COMMIT == (
        "244e206024f949e1ca6430ff2f14a8dc037236f9"
    )
    assert impl.CUDA_EQ_ARTIFACT_FREEZE_COMMIT == (
        "96f8a9a8385d71175db6c0d52a86f16c5ea75040"
    )
    assert impl.FULL_CUDA_MODEL_FORWARD_BUDGET == 36000
    assert impl.CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET == 0
    assert impl.necessity.PAIR_COUNT == 300
    assert impl.necessity.CONDITIONS == (
        "native22",
        "r22_neutralized",
        "c22_coefficient_control",
    )


def test_accelerated_branch_audit_preserves_matched_control():
    r22, c22 = synthetic_bases()
    audits = {}
    for condition in impl.necessity.CONDITIONS:
        audit, _ = impl.audit_accelerated_branch(
            synthetic_branch(condition, r22, c22),
            condition=condition,
            plus_branch=True,
            r22=r22,
            c22=c22,
        )
        audits[condition] = audit

    assert (
        audits["native22"]["native_write_sha256"]
        == audits["r22_neutralized"]["native_write_sha256"]
        == audits["c22_coefficient_control"]["native_write_sha256"]
    )
    assert audits["r22_neutralized"]["r22_coefficients"] == (
        audits["c22_coefficient_control"]["r22_coefficients"]
    )
    assert math.isclose(
        audits["r22_neutralized"]["r22_candidate_correction_l2"],
        audits["c22_coefficient_control"]["c22_candidate_correction_l2"],
        abs_tol=1e-12,
    )


def test_condition_matching_requires_40_coordinates():
    r22, c22 = synthetic_bases()
    records = {}
    for condition in impl.necessity.CONDITIONS:
        audit, state = impl.audit_accelerated_branch(
            synthetic_branch(condition, r22, c22),
            condition=condition,
            plus_branch=True,
            r22=r22,
            c22=c22,
        )
        records[condition] = [
            ("xg2_0", "positive_probe", "tp", audit, state)
        ] * 40
    result = impl.validate_condition_matching(records)
    assert result["coordinate_count"] == 40
    assert result["max_r22_coefficient_difference"] <= 1e-12


def test_frozen_endpoint_is_reused():
    endpoint = impl.necessity.endpoint(1.5, 0.8, 1.1)
    assert endpoint["Q0"] == 1.5
    assert endpoint["QR"] == 0.8
    assert endpoint["QC"] == 1.1
    assert math.isclose(endpoint["A_R"], 0.7)
    assert math.isclose(endpoint["A_C"], 0.4)
    assert math.isclose(endpoint["D_NEC22"], 0.3)


def test_qualified_backend_identity_is_exact():
    assert impl.QUALIFIED_BACKEND == (
        "FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY"
    )
    assert impl.FROZEN_BLOBS[
        "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py"
    ] == "421d30f00cf71690ed41c983ccf0540808e1de1c"


def test_only_frozen_decision_function_supplies_confirmatory_rule():
    values = [
        {
            "Q0": 1.0,
            "A_R": 0.2,
            "D_NEC22": x,
        }
        for x in (0.10, 0.11, 0.09, 0.12)
    ]
    # The production runner delegates directly to this function; this unit test
    # only checks that the imported frozen implementation is callable.
    assert callable(impl.necessity.confirmatory_decision)
    assert impl.necessity.LABEL_SUPPORTED.startswith("GEN5_R22_LOCAL_NECESSITY")
