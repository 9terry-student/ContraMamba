from __future__ import annotations

import inspect
import math

import torch
from torch import nn

from scripts import (
    reason_router_gen4_mamba370m14b_stagewise_coupling_localization_fast_cuda
    as runner,
)


def test_protocol_is_fixed_and_descriptive() -> None:
    assert runner.SCALE_ORDER == ("mamba370m", "mamba14b")
    assert runner.CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_control",
    )
    assert runner.STAGE_ORDER[0] == "pre_block_35"
    assert runner.STAGE_ORDER[1] == "post_block_35"
    assert runner.STAGE_ORDER[-2] == "post_block_47"
    assert runner.STAGE_ORDER[-1] == "post_final_norm"
    assert len(runner.STAGE_ORDER) == 15
    assert runner.FULL_FORWARDS_PER_SCALE == 1800
    assert runner.DOWNSTREAM_REPLAYS_PER_SCALE == 3600


def test_runner_has_no_inference_or_selection_path() -> None:
    source = inspect.getsource(runner).lower()
    for token in ("scipy", "ttest_1samp", "confirmation_inference.json"):
        assert token not in source
    assert "rescue_performed" in source
    assert "selection_reopened" in source


def test_metrics_from_output() -> None:
    output = {
        "logits": torch.tensor([[1.0, 2.0, 0.5]]),
        "frame_prob": torch.tensor([0.8]),
        "predicate_coverage_prob": torch.tensor([0.7]),
        "sufficiency_prob": torch.tensor([0.6]),
        "positive_energy": torch.tensor([0.4]),
        "negative_energy": torch.tensor([0.3]),
        "q_authorized": torch.tensor([0.336]),
        "entitlement_prob": torch.tensor([0.336]),
    }
    got = runner.metrics_from_output(output, label_id=1)
    assert got["prediction_id"] == 1
    assert math.isclose(got["correct_class_logit_margin"], 1.0)
    assert math.isclose(got["frame_prob"], 0.8, rel_tol=1e-6)


class DummyBlock(nn.Module):
    def forward(self, x):
        return x + 1.0


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock() for _ in range(48)])
        self.norm_f = nn.Identity()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = DummyBackbone()


def test_capture_hooks_observe_without_mutating() -> None:
    model = DummyModel()
    captures = {}
    handles = runner.install_capture_hooks(model, captures)
    x = torch.zeros(1, 3, 4)
    y = x
    for block in model.mamba.layers:
        y = block(y)
    y = model.mamba.norm_f(y)
    runner.remove_handles(handles)

    assert tuple(captures) == runner.STAGE_ORDER
    assert torch.equal(captures["pre_block_35"], torch.full_like(x, 35.0))
    assert torch.equal(captures["post_block_35"], torch.full_like(x, 36.0))
    assert torch.equal(captures["post_block_47"], torch.full_like(x, 48.0))
    assert torch.equal(captures["post_final_norm"], torch.full_like(x, 48.0))
    assert torch.equal(y, torch.full_like(x, 48.0))


def test_propagation_stats_are_causal_prefix_aware() -> None:
    a = torch.zeros(1, 5, 2)
    b = torch.zeros(1, 5, 2)
    a[0, 2:, 0] = 1.0
    got = runner.propagation_stats(
        a, b,
        target_index=2,
        attended_length=5,
    )
    assert got["pre_target_max_abs"] == 0.0
    assert math.isclose(got["target_token_l2"], 1.0)
    assert math.isclose(
        got["attended_suffix_l2"],
        math.sqrt(3.0),
        rel_tol=1.0e-6,
        abs_tol=1.0e-7,
    )


def test_full_forward_budget_is_smaller_than_old_four_condition_bridge() -> None:
    assert runner.FULL_FORWARDS_PER_SCALE == 1800
    assert runner.bridge.TOTAL_FORWARD_BUDGET == 2400
