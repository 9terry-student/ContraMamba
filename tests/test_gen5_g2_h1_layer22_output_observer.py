"""CPU-only synthetic numerical and lifecycle gates for Gen5 G2-H1 output observer.

No pretrained backbone, task dataset, checkpoint, training or GPU.
Install in: tests/test_gen5_g2_h1_layer22_output_observer.py
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from contramamba.gen5_g2_h1_layer22_output_observer import (
    Layer22OutputObserver,
    ObservationContractError,
    final_logit_rs_contrasts,
    masked_token_statistics,
)


class TinyNative(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor(
                [[0.5, 0.1], [-0.2, 0.25]], dtype=torch.float64))
            self.proj.bias.copy_(torch.tensor(
                [0.15, -0.30], dtype=torch.float64))
        self.seen_masks = []
        self.calls = 0

    def forward(self, hidden_states, cache_params=None, cache_position=None,
                attention_mask=None):
        assert cache_params is None and cache_position is None
        self.seen_masks.append(attention_mask)
        self.calls += 1
        y = self.proj(hidden_states)
        if attention_mask is not None:
            # This makes a mask forwarded to native observably different.
            y = y + attention_mask.to(y.dtype).unsqueeze(-1)
        return y


class TinyG2Wrapper(nn.Module):
    """Toy preserving G2 order and out-of-band effective-mask semantics."""

    def __init__(self):
        super().__init__()
        self.native_mixer = TinyNative()
        self._phase2_active_mask = None
        self.correction_scale = 0.37

    @contextmanager
    def active_mask(self, mask):
        assert self._phase2_active_mask is None
        self._phase2_active_mask = mask
        try:
            yield
        finally:
            self._phase2_active_mask = None

    def correction_output(self, hidden_states, effective_mask):
        # This is NOT intended to reproduce the Mamba recurrent write.
        a = hidden_states * self.correction_scale
        b = torch.stack(
            [a[..., 0] + 0.2 * a[..., 1],
             -0.3 * a[..., 0] + a[..., 1]], dim=-1)
        return b * effective_mask.to(b.dtype).unsqueeze(-1)

    def forward(self, hidden_states, cache_params=None, cache_position=None,
                attention_mask=None):
        native = self.native_mixer(
            hidden_states, cache_params=cache_params,
            cache_position=cache_position, attention_mask=attention_mask)
        effective = (attention_mask if attention_mask is not None
                     else self._phase2_active_mask)
        if effective is None:
            raise ValueError("effective mask required")
        return native + self.correction_output(hidden_states, effective)


@pytest.fixture
def small():
    x = torch.tensor([
        [[1., 2.], [3., 4.], [5., 6.], [900., 800.]],
        [[2., 1.], [4., 3.], [6., 5.], [700., 800.]],
        [[7., 1.], [8., 2.], [9., 3.], [600., 900.]],
        [[1., 7.], [2., 8.], [3., 9.], [850., 700.]],
    ], dtype=torch.float64)
    valid = torch.tensor([
        [1, 1, 1, 0], [1, 1, 0, 0],
        [1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    claim = torch.tensor([
        [1, 0, 0, 0], [1, 0, 0, 0],
        [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
    evidence = torch.tensor([
        [0, 0, 1, 0], [0, 1, 0, 0],
        [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.long)
    # Intentionally non-adjacent pair cells. Never pair by row position.
    pairs = ["p1", "p2", "p1", "p2"]
    cells = ["C6_EXPLICIT_DENIAL", "C0_SHAM",
             "C0_SHAM", "C6_EXPLICIT_DENIAL"]
    return x, valid, claim, evidence, pairs, cells


def _observer(wrapper, small):
    _x, valid, claim, evidence, pairs, cells = small
    return Layer22OutputObserver(
        wrapper, attention_mask=valid, claim_mask=claim,
        evidence_mask=evidence, pair_ids=pairs, cell_ids=cells,
        row_ids=["a", "b", "c", "d"])


def _manual_stat(tensor, mask):
    # Intentionally independent indexing/arithmetic implementation.
    vals = [
        float(tensor[i, d].item())
        for i in range(tensor.shape[0]) if bool(mask[i])
        for d in range(tensor.shape[1])
    ]
    if not vals:
        return (0, None, None, None)
    n = len(vals)
    return (int(mask.sum().item()), sum(vals) / n,
            sum(abs(v) for v in vals) / n,
            math.sqrt(sum(v*v for v in vals) / n))


def test_forward_noop_equivalence_and_native_mask_identity(small):
    x, valid, claim, evidence, pairs, cells = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    orig_native_hook_count = len(wrapper.native_mixer._forward_hooks)
    orig_outer_hook_count = len(wrapper._forward_hooks)
    head = nn.Linear(2, 3, bias=False, dtype=torch.float64).eval()
    with torch.no_grad():
        head.weight.copy_(torch.tensor(
            [[1., -0.5], [-0.25, 0.1], [0.5, 0.75]], dtype=torch.float64))
    with torch.inference_mode(), wrapper.active_mask(valid):
        baseline = wrapper(x)    # Native attention_mask is None on purpose.
        baseline_logits = head((baseline * valid.unsqueeze(-1)).sum(dim=1))
    frozen_parameters = [p.detach().clone()
                         for p in tuple(wrapper.parameters()) +
                         tuple(head.parameters())]
    assert wrapper.native_mixer.seen_masks == [None]

    obs = _observer(wrapper, small)
    with torch.inference_mode(), wrapper.active_mask(valid), obs:
        hooked = wrapper(x)
        hooked_logits = head((hooked * valid.unsqueeze(-1)).sum(dim=1))
    assert torch.equal(baseline, hooked)
    assert torch.equal(baseline_logits, hooked_logits)
    assert all(torch.equal(x, y) for x, y in zip(
        frozen_parameters, tuple(wrapper.parameters()) +
        tuple(head.parameters())))
    assert wrapper.native_mixer.seen_masks == [None, None]
    assert wrapper.native_mixer.calls == 2
    assert wrapper._phase2_active_mask is None
    assert len(wrapper.native_mixer._forward_hooks) == orig_native_hook_count
    assert len(wrapper._forward_hooks) == orig_outer_hook_count

    record = obs.record()
    torch.testing.assert_close(record.native, wrapper.native_mixer.proj(x),
                               rtol=0, atol=0)
    torch.testing.assert_close(record.wrapped, baseline, rtol=0, atol=0)
    torch.testing.assert_close(
        record.contribution, wrapper.correction_output(x, valid),
        rtol=0, atol=1e-13)
    assert not record.contribution.requires_grad
    assert record.native.data_ptr() != record.wrapped.data_ptr()
    assert set(record.pair_differences) == {"p1", "p2"}
    assert record.per_row[1]["contribution"]["evidence"]["tokens"] == 1
    assert record.per_row[1]["contribution"]["other"]["tokens"] == 0


def test_masked_aggregation_matches_independent_loop(small):
    x, valid, claim, evidence, pairs, cells = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    with torch.inference_mode(), wrapper.active_mask(valid), _observer(wrapper, small) as obs:
        wrapper(x)
    record = obs.record()
    for row in range(4):
        masks = {
            "valid": valid[row].bool(),
            "claim": valid[row].bool() & claim[row].bool(),
            "evidence": valid[row].bool() & evidence[row].bool(),
            "other": valid[row].bool() & ~(
                claim[row].bool() | evidence[row].bool()),
        }
        for source, tensor in (
            ("native", record.native),
            ("wrapped", record.wrapped),
            ("contribution", record.contribution),
        ):
            for region, mask in masks.items():
                count, mean, ma, rms = _manual_stat(tensor[row], mask)
                observed = record.per_row[row][source][region]
                assert observed["tokens"] == count
                for key, expected in (
                    ("mean", mean), ("mean_abs", ma), ("rms", rms),
                ):
                    if expected is None:
                        assert observed[key] is None
                    else:
                        assert observed[key] == pytest.approx(
                            expected, abs=1e-12)
    # A signed mean, not a norm, must be reported as a separate observable.
    assert record.per_row[0]["contribution"]["valid"]["mean"] != (
        record.per_row[0]["contribution"]["valid"]["rms"])


def test_padding_excluded_even_when_extremely_large(small):
    x, valid, claim, evidence, pairs, cells = small
    with torch.no_grad():
        wrapper = TinyG2Wrapper().eval().requires_grad_(False)
        with wrapper.active_mask(valid), _observer(wrapper, small) as o1:
            wrapper(x)
        y = x.clone()
        y[~valid.bool()] = 1e12
        with wrapper.active_mask(valid), _observer(wrapper, small) as o2:
            wrapper(y)
    a, b = o1.record(), o2.record()
    for i in range(4):
        assert a.per_row[i] == b.per_row[i]
    # Padded native values can differ; only masked summaries are invariant.
    assert not torch.equal(a.native, b.native)


def test_pair_contrasts_use_identity_not_row_adjacency(small):
    x, valid, claim, evidence, pairs, cells = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    with torch.no_grad(), wrapper.active_mask(valid), _observer(wrapper, small) as o:
        wrapper(x)
    record = o.record()
    for pair, c6, c0 in (("p1", 0, 2), ("p2", 3, 1)):
        for source in ("native", "wrapped", "contribution"):
            for region in ("valid", "claim", "evidence", "other"):
                for field in ("mean", "mean_abs", "rms"):
                    a = record.per_row[c0][source][region][field]
                    b = record.per_row[c6][source][region][field]
                    observed = record.pair_differences[pair][source][region][
                        field + "_C6_minus_C0"]
                    if a is None or b is None:
                        assert observed is None
                    else:
                        assert observed == pytest.approx(b-a, abs=1e-12)

    logits = torch.tensor([
        [2., 0., 0.], [0., 0., 1.],
        [0., 0., 2.], [3., 0., 1.],
    ], dtype=torch.float64)
    assert final_logit_rs_contrasts(logits, pairs, cells) == {
        "p1": 4.0, "p2": 3.0,
    }


def test_duplicate_nested_observer_blocked_and_hooks_cleaned(small):
    x, valid, *_ = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    first, second = _observer(wrapper, small), _observer(wrapper, small)
    with first:
        with pytest.raises(ObservationContractError,
                           match="DUPLICATE_OBSERVER_REGISTRATION"):
            with second:
                raise AssertionError("must not enter")
        with torch.inference_mode(), wrapper.active_mask(valid):
            wrapper(x)
    assert first.record().wrapped.shape == x.shape
    assert not wrapper._forward_hooks
    assert not wrapper.native_mixer._forward_hooks


def test_two_forwards_detected_and_cleanup_occurs(small):
    x, valid, *_ = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    obs = _observer(wrapper, small)
    with pytest.raises(ObservationContractError, match="NATIVE_CALLED_MULTIPLE"):
        with torch.no_grad(), wrapper.active_mask(valid), obs:
            wrapper(x)
            wrapper(x)
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks
    with pytest.raises(ObservationContractError, match="NOT_COMPLETE"):
        obs.record()


def test_exception_and_missing_forward_cleaned(small):
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    obs = _observer(wrapper, small)
    with pytest.raises(ValueError, match="deliberate"):
        with obs:
            raise ValueError("deliberate")
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks
    with pytest.raises(ObservationContractError, match="NOT_COMPLETE"):
        obs.record()
    with pytest.raises(ObservationContractError, match="EXPECTED_EXACTLY_ONE"):
        with _observer(wrapper, small):
            pass
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks


@pytest.mark.parametrize("mutator,reason", [
    (lambda s: s[2].__setitem__((0, 3), 1), "CLAIM_IN_PADDING"),
    (lambda s: s[1].__setitem__((0, 0), 2), "ATTENTION_MASK_NON_BINARY"),
])
def test_bad_masks_fail_closed(small, mutator, reason):
    items = list(small)
    items[1], items[2], items[3] = (
        items[1].clone(), items[2].clone(), items[3].clone())
    mutator(items)
    with pytest.raises(ObservationContractError, match=reason):
        _observer(TinyG2Wrapper().eval().requires_grad_(False), tuple(items))


def test_duplicate_pair_cell_and_wrong_logits_shape_rejected(small):
    x, valid, claim, evidence, pairs, cells = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    with pytest.raises(ObservationContractError, match="DUPLICATE_PAIR_CELL"):
        Layer22OutputObserver(
            wrapper, attention_mask=valid, claim_mask=claim,
            evidence_mask=evidence, pair_ids=["p1"]*4,
            cell_ids=["C0_SHAM", "C0_SHAM", "C6_EXPLICIT_DENIAL", "C1_TITLE"])
    with pytest.raises(ObservationContractError, match="LOGITS_SHAPE"):
        final_logit_rs_contrasts(torch.ones(4, 2), pairs, cells)


def test_invalid_output_shape_and_eval_gate(small):
    x, valid, *_ = small
    wrapper = TinyG2Wrapper().train()
    with pytest.raises(ObservationContractError, match="EVAL_ONLY"):
        _observer(wrapper, small)
    wrapper.eval().requires_grad_(False)
    with pytest.raises(ObservationContractError, match="REQUIRES_GRAD"):
        # Force autograd output with a floating input tensor that needs grad.
        with wrapper.active_mask(valid), _observer(wrapper, small):
            wrapper(x.clone().requires_grad_(True))
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks


def test_overlap_claim_evidence_semantics_independent():
    value = torch.tensor([[[2., -2.], [3., 9.], [999., 999.]]])
    valid = torch.tensor([[1, 1, 0]])
    claim = torch.tensor([[1, 1, 0]])
    evidence = torch.tensor([[0, 1, 0]])
    s = masked_token_statistics(value, valid, claim, evidence)[0]
    assert s["claim"]["tokens"] == 2
    assert s["evidence"]["tokens"] == 1
    assert s["other"]["tokens"] == 0
    assert s["other"]["mean"] is None


def test_explicit_mask_preserved_and_observer_effective_mask_gate(small):
    x, valid, *_ = small
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    obs = _observer(wrapper, small)
    with torch.inference_mode(), obs:
        wrapper(x, attention_mask=valid)
    assert wrapper.native_mixer.seen_masks[0] is valid
    assert obs.record().wrapped.shape == x.shape
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks

    wrong = valid.clone()
    wrong[0, 1] = 0
    with pytest.raises(ObservationContractError,
                       match="OBSERVER_EFFECTIVE_MASK_MISMATCH"):
        with torch.inference_mode(), wrapper.active_mask(wrong), _observer(wrapper, small):
            wrapper(x)
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks


def test_native_mask_semantics_violation_detected(small):
    x, valid, *_ = small

    class WrongNativeMask(TinyG2Wrapper):
        def forward(self, hidden_states, cache_params=None,
                    cache_position=None, attention_mask=None):
            effective = (attention_mask if attention_mask is not None
                         else self._phase2_active_mask)
            # Forbidden: out-of-band effective mask injected into native.
            native = self.native_mixer(hidden_states, attention_mask=effective)
            return native + self.correction_output(hidden_states, effective)

    wrapper = WrongNativeMask().eval().requires_grad_(False)
    with pytest.raises(ObservationContractError,
                       match="NATIVE_ATTENTION_MASK_FORWARDING_CHANGED"):
        with torch.no_grad(), wrapper.active_mask(valid), _observer(wrapper, small):
            wrapper(x)
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks
