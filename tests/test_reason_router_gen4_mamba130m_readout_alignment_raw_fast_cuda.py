from __future__ import annotations

import inspect

import torch

from scripts import (
    reason_router_gen4_mamba130m_readout_alignment_raw_fast_cuda
    as subject,
)


def test_protocol_is_exact() -> None:
    assert subject.PAIR_IDS[0] == "xg1_fact_2701"
    assert subject.PAIR_IDS[-1] == "xg1_fact_3000"
    assert subject.PAIR_COUNT == 300
    assert subject.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert subject.SHARD_RANGES == {
        0: (2701, 2850),
        1: (2851, 3000),
    }
    assert subject.ROWS_PER_SHARD == 300
    assert subject.TOTAL_ROWS == 600
    assert subject.SCIENTIFIC_FORWARD_COUNT == 600
    assert subject.LOCAL_BACKWARD_COUNT == 600
    assert subject.SELECTED_PLANE == "P3"
    assert subject.CONTROL_PLANE == "P5"
    assert subject.INTERVENTION_LAYER == 17
    assert subject.TARGET_OFFSET == 2
    assert subject.DIM == 395
    assert subject.PRIMARY_ENDPOINT == "Delta_L_130M"
    assert subject.PRIMARY_TEST == "one_sample_student_t_greater"
    assert subject.PRIMARY_P_VALUE_COUNT == 1


def test_checkpoint_and_geometry_are_frozen_seed181_identity() -> None:
    assert subject.CHECKPOINT_SHA256 == (
        "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
    )
    assert subject.GEOMETRY_JSON_SHA256 == (
        "e6e9db909eb7d2c6bbdb493a4efeca8c18e4d474cf99a943be0f3f7b9dee1012"
    )
    assert subject.GEOMETRY_PT_SHA256 == (
        "de3ae6a450c2ba0a85b4f53919e3765e6e7dfb6dba3676535554c437b1647a1c"
    )


def test_shards_exactly_partition_pairs() -> None:
    s0 = subject.shard_pairs(0)
    s1 = subject.shard_pairs(1)
    assert len(s0) == len(s1) == 150
    assert s0[0] == "xg1_fact_2701"
    assert s0[-1] == "xg1_fact_2850"
    assert s1[0] == "xg1_fact_2851"
    assert s1[-1] == "xg1_fact_3000"
    assert s0 + s1 == subject.PAIR_IDS


def test_local_leaf_hook_cuts_upstream_and_preserves_values(monkeypatch) -> None:
    monkeypatch.setattr(subject, "DIM", 2)

    class Mixer:
        def __init__(self):
            self.in_proj = torch.nn.Identity()

    mixer = Mixer()
    capture = {}
    mask = torch.tensor([True, False, True, False])
    handle = subject._install_local_leaf_hook(
        mixer,
        token_index=1,
        intermediate_size=4,
        strong_mask=mask,
        capture=capture,
    )
    try:
        x = torch.arange(24.0).view(1, 3, 8).requires_grad_(True)
        y = mixer.in_proj(x)
        assert torch.equal(y.detach(), x.detach())
        loss = y[0, 1, :4].sum()
        grad = torch.autograd.grad(loss, capture["leaf"])[0]
        assert torch.equal(grad, torch.ones(4))
        assert x.grad is None
    finally:
        handle.remove()


def test_active_margin_exact() -> None:
    logits = torch.tensor([[1.0, 2.0, 5.0]], requires_grad=True)
    margin, wrong = subject._active_margin(logits, 2)
    assert wrong == 1
    assert margin.item() == 3.0


def test_plane_components_use_p3_coefficients_for_p5_control(monkeypatch) -> None:
    monkeypatch.setattr(subject, "DIM", 4)

    planes = {
        "pp3_plus": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "pp3_minus": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "pp5_plus": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "pp5_minus": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }

    class Restoration:
        TOL = 1e-12

    monkeypatch.setattr(subject.bridge, "restoration", Restoration())

    h = torch.tensor([3.0, 4.0, 9.0, 8.0])
    out = subject._plane_components(h, planes=planes)
    assert out["a"] == 3.0
    assert out["b"] == 4.0
    assert torch.equal(
        out["selected_component"],
        torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert torch.equal(
        out["control_component"],
        torch.tensor([0.0, 0.0, 3.0, 4.0], dtype=torch.float64),
    )


def test_raw_source_contains_no_statistical_inference_or_behavior_merge() -> None:
    src = inspect.getsource(subject).lower()
    assert "scipy" not in src
    assert "ttest" not in src
    assert '"p_value_count_executed": 0' in src
    assert '"behavioral_bridge_d_beh_accessed": false' in src
    assert '"cross_scale_readout_result_accessed": false' in src


def test_raw_contract_has_no_intervention_condition_forward() -> None:
    src = inspect.getsource(subject.run_raw)
    assert '"intervention_condition_forward_count": 0' in src
    assert '"scientific_full_model_forward_count": SCIENTIFIC_FORWARD_COUNT' in src
    assert '"local_backward_count": LOCAL_BACKWARD_COUNT' in src


def test_gate_discards_numeric_values() -> None:
    src = inspect.getsource(subject.run_technical_gate)
    assert '"NUMERIC_MARGIN_RETAINED=False"' in src
    assert '"NUMERIC_GRADIENT_RETAINED=False"' in src
    assert '"NUMERIC_ALIGNMENT_RETAINED=False"' in src
