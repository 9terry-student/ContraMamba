"""Synthetic-only tests for frozen six-arm diagnostic companion; NO real model forward."""
import hashlib
import math

import pytest
import torch
from torch import nn

from contramamba import gen5_shared_bypass_functional_core_diagnostic as target


class FakeMixer(nn.Module):
    hidden_size = 768
    intermediate_size = 1536
    ssm_state_size = 16
    layer_idx = 22

    def __init__(self):
        super().__init__()
        self.register_buffer('parent_identity', torch.tensor([0.8]))
        self.calls = 0
        self.received_attention_mask = None

    def forward(self, hidden_states, cache_params=None, cache_position=None,
                attention_mask=None):
        assert cache_params is None and cache_position is None
        self.calls += 1
        self.received_attention_mask = attention_mask
        return hidden_states * self.parent_identity


class TinyCorrection(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.arm = 'G5-C0'
        self.A_theta = nn.Linear(768, 2, bias=False)
        self.B_theta = nn.Linear(2, 24576, bias=False)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.A_theta.weight.copy_(torch.randn((2, 768), generator=g) * .001)
            self.B_theta.weight.copy_(torch.randn((24576, 2), generator=g) * .001)
        self.requires_grad_(False)
        self.eval()


def fake_accelerated(mixer, hidden_states, correction, *, attention_mask,
                     checkpoint_recompute):
    assert checkpoint_recompute is False
    a = correction.A_theta.weight
    b = correction.B_theta.weight[:768]
    latent = torch.nn.functional.linear(hidden_states, a)
    output = torch.nn.functional.linear(latent, b)
    return output * attention_mask.unsqueeze(-1)


@pytest.fixture
def sources(monkeypatch):
    correction = {seed: TinyCorrection(seed) for seed in target.SEEDS}
    identity = {}
    for seed, c in correction.items():
        identity[seed] = {
            'A_theta.weight': target._tensor_sha256(c.A_theta.weight),
            'B_theta.weight': target._tensor_sha256(c.B_theta.weight),
        }
    monkeypatch.setattr(target, 'EXPECTED_SOURCE_TENSOR_SHA256', identity)
    monkeypatch.setattr(target, 'accelerated_correction_mixer_contribution', fake_accelerated)
    return correction


def test_all_18_static_frozen_arms_and_linear_superposition(sources):
    target.validate_frozen_sources(sources)
    x = torch.randn((1, 3, 768), generator=torch.Generator().manual_seed(6))
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    for heldout in target.SEEDS:
        outs = {}
        for arm in target.ARMS:
            mixer = FakeMixer().eval()
            wrapper = target.FrozenSharedBypassDiagnosticMixer(
                native_mixer=mixer, source_corrections=sources,
                heldout_seed=heldout, arm=arm).eval()
            with torch.no_grad():
                with wrapper.active_mask(mask):
                    outs[arm] = wrapper(x)
            assert mixer.calls == 1
            assert mixer.received_attention_mask is None
            assert wrapper._phase2_active_mask is None
            assert tuple(outs[arm].shape) == tuple(x.shape)
            assert bool(torch.isfinite(outs[arm]).all())
            assert wrapper.arm == arm
        native = x * .8
        torch.testing.assert_close(outs['ZERO'], native, rtol=0, atol=1e-7)
        expected_full = native + fake_accelerated(
            None, x, sources[heldout], attention_mask=mask,
            checkpoint_recompute=False,
        )
        torch.testing.assert_close(outs['FULL'], expected_full, rtol=1e-5, atol=1e-6)
        reconstructed = outs['SHARED_ONLY'] + outs['RESIDUAL_ONLY'] - native
        torch.testing.assert_close(reconstructed, outs['FULL'], rtol=1e-5, atol=1e-6)
        assert set(target.FROZEN_ARM_COEFFICIENTS[heldout]) == set(target.ARMS)


def test_source_identity_and_no_training_guard(sources):
    sources[6201].B_theta.weight.requires_grad_(True)
    with pytest.raises(target.DiagnosticContractError, match='SOURCE_TRAINABLE'):
        target.validate_frozen_sources(sources)
    sources[6201].B_theta.weight.requires_grad_(False)
    with torch.no_grad():
        sources[6201].B_theta.weight[0, 0] += 1.
    with pytest.raises(target.DiagnosticContractError, match='SOURCE_B_IDENTITY'):
        target.validate_frozen_sources(sources)


def test_eval_only_and_non_authorized_arms(sources):
    wrapper = target.FrozenSharedBypassDiagnosticMixer(
        native_mixer=FakeMixer().eval(), source_corrections=sources,
        heldout_seed=6201, arm='ZERO',
    ).eval()
    with pytest.raises(target.DiagnosticContractError, match='DIAGNOSTIC_TRAINING_FORBIDDEN'):
        wrapper.train(True)
    with pytest.raises(target.DiagnosticContractError, match='GRAD_ENABLED_FORBIDDEN'):
        wrapper(torch.zeros((1, 1, 768)), attention_mask=torch.ones((1, 1)))
    with pytest.raises(target.DiagnosticContractError, match='UNAUTHORIZED_ARM'):
        target.select_frozen_coefficients(6201, 'NEW_ARM')
    with pytest.raises(target.DiagnosticContractError, match='UNAUTHORIZED_HELDOUT_SEED'):
        target.select_frozen_coefficients(6204, 'FULL')


def test_active_mask_contract_fails_closed(sources):
    wrapper = target.FrozenSharedBypassDiagnosticMixer(
        native_mixer=FakeMixer().eval(), source_corrections=sources,
        heldout_seed=6202, arm='SHARED_ONLY',
    ).eval()
    x = torch.zeros((1, 2, 768))
    with torch.no_grad():
        with pytest.raises(target.DiagnosticContractError, match='ACTIVE_MASK_REQUIRED'):
            wrapper(x)
        with pytest.raises(target.DiagnosticContractError, match='ACTIVE_MASK_SHAPE'):
            wrapper(x, attention_mask=torch.ones((1, 3)))
        with pytest.raises(target.DiagnosticContractError, match='CACHE_FORBIDDEN'):
            wrapper(x, cache_params=object(), attention_mask=torch.ones((1, 2)))
