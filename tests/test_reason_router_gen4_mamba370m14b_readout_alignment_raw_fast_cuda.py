from __future__ import annotations

import inspect
import torch

from scripts import (
    reason_router_gen4_mamba370m14b_readout_alignment_raw_fast_cuda as subject,
)

def test_protocol_is_exact() -> None:
    assert subject.PAIR_COUNT == 300
    assert subject.ROWS_PER_SCALE == 600
    assert subject.TOTAL_ROWS == 1200
    assert subject.PRIMARY_ENDPOINT == "R=Delta_L_370M-Delta_L_1.4B"
    assert subject.PRIMARY_TEST == "paired_one_sample_student_t_greater"
    assert subject.PRIMARY_P_VALUE_COUNT == 1
    assert subject.SCALE_TO_PHYSICAL_GPU == {
        "mamba370m": 0,
        "mamba14b": 1,
    }

def test_local_leaf_hook_cuts_upstream_and_preserves_values() -> None:
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
        g = torch.autograd.grad(loss, capture["leaf"])[0]
        assert torch.equal(g, torch.ones(4))
        assert x.grad is None
    finally:
        handle.remove()

def test_active_margin_exact() -> None:
    logits = torch.tensor([[1.0, 2.0, 5.0]], requires_grad=True)
    margin, wrong = subject._active_margin(logits, 2)
    assert wrong == 1
    assert margin.item() == 3.0

def test_raw_source_contains_no_statistical_inference() -> None:
    src = inspect.getsource(subject).lower()
    assert "scipy" not in src
    assert "ttest" not in src
    assert '"inferential_test_performed": false' in src
    assert '"experiment1_d_beh_accessed": false' in src

def test_manifold_capture_adds_no_forward_or_backward() -> None:
    src = inspect.getsource(subject.run_raw)
    assert '"additional_model_forward_count": 0' in src
    assert '"additional_backward_count": 0' in src
