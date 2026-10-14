"""CPU-only G1 synthetic-parent installation, rollback, and mask tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contramamba import gen5_shared_bypass_functional_core_diagnostic as companion
from contramamba.gen5_phase2_state_update_ownership import (
    parent_parameter_fingerprint,
    phase2_active_mask,
)
from contramamba.gen5_shared_bypass_g1_installation_diagnostic import (
    G1Installation,
    DiagnosticContractError,
    _native_buffer_sha256,
    g1_companion_active_mask,
    install_g1_synthetic_diagnostic,
    installed_g1_synthetic_diagnostic,
    remove_g1_synthetic_diagnostic,
)


class FakeNativeMixer(nn.Module):
    hidden_size = 768
    intermediate_size = 1536
    ssm_state_size = 16
    layer_idx = 22

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.375), requires_grad=False)
        self.register_buffer("marker", torch.tensor([0.12]))
        self.masks = []

    def forward(self, hidden_states, cache_params=None, cache_position=None,
                attention_mask=None):
        assert cache_params is None and cache_position is None
        self.masks.append(attention_mask)
        return self.scale * hidden_states


class TinyC0(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.arm = "G5-C0"
        self.A_theta = nn.Linear(768, 2, bias=False)
        self.B_theta = nn.Linear(2, 24576, bias=False)
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.A_theta.weight.copy_(torch.randn((2, 768), generator=g) * 0.001)
            self.B_theta.weight.copy_(torch.randn((24576, 2), generator=g) * 0.001)
        self.requires_grad_(False)
        self.eval()


class FakeBlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.mixer = FakeNativeMixer() if idx == 22 else nn.Identity()


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in range(24))
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids):
        # A synthetic route to mixer 22, not a pretrained Mamba forward.
        return self.layers[22].mixer(input_ids)


class FakeParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = FakeBackbone()
        self.requires_grad_(False)
        self.eval()


def fake_correction(native_mixer, hidden_states, correction, *,
                    attention_mask, checkpoint_recompute):
    assert checkpoint_recompute is False
    latent = F.linear(hidden_states, correction.A_theta.weight)
    out = F.linear(latent, correction.B_theta.weight[:768])
    return out * attention_mask.unsqueeze(-1)


@pytest.fixture
def fixture(monkeypatch):
    model = FakeParent()
    sources = {seed: TinyC0(seed) for seed in companion.SEEDS}
    hashes = {
        seed: {
            "A_theta.weight": companion._tensor_sha256(source.A_theta.weight),
            "B_theta.weight": companion._tensor_sha256(source.B_theta.weight),
        } for seed, source in sources.items()
    }
    monkeypatch.setattr(companion, "EXPECTED_SOURCE_TENSOR_SHA256", hashes)
    monkeypatch.setattr(companion, "accelerated_correction_mixer_contribution", fake_correction)
    return model, sources


def test_g1_install_mask_restore_fingerprint_and_layer_identity(fixture):
    model, sources = fixture
    original_mixer = model.mamba.layers[22].mixer
    ids = tuple(id(layer.mixer) for layer in model.mamba.layers)
    before_params = parent_parameter_fingerprint(model)
    before_buffers = _native_buffer_sha256(model)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    inputs = torch.ones((1, 3, 768))
    with installed_g1_synthetic_diagnostic(
        model, source_corrections=sources, heldout_seed=6201, arm="FULL"
    ) as installed:
        assert model.mamba.layers[22].mixer is installed
        assert model.mamba.layers[22].mixer.native_mixer is original_mixer
        assert not any(p.requires_grad for p in model.parameters())
        assert tuple(id(x.mixer) for i, x in enumerate(model.mamba.layers) if i != 22) == \
               tuple(v for i, v in enumerate(ids) if i != 22)
        with torch.no_grad():
            with g1_companion_active_mask(model, mask):
                y = model.mamba(input_ids=inputs)
        assert y.shape == inputs.shape and torch.isfinite(y).all()
        assert original_mixer.masks[-1] is None
        assert parent_parameter_fingerprint(model) == before_params
        assert _native_buffer_sha256(model) == before_buffers
        assert installed._phase2_active_mask is None
        # Old Phase2 helper must not silently accept a companion.
        with pytest.raises(Exception, match="LAYER22_NOT_WRAPPED"):
            with phase2_active_mask(model, mask):
                pass
    assert model.mamba.layers[22].mixer is original_mixer
    assert tuple(id(layer.mixer) for layer in model.mamba.layers) == ids
    assert parent_parameter_fingerprint(model) == before_params
    assert _native_buffer_sha256(model) == before_buffers
    with pytest.raises(DiagnosticContractError, match="G1_COMPANION_NOT_INSTALLED"):
        with g1_companion_active_mask(model, mask):
            pass


def test_g1_duplicate_and_nested_wrapper_rejected(fixture):
    model, sources = fixture
    original = model.mamba.layers[22].mixer
    h = install_g1_synthetic_diagnostic(
        model, source_corrections=sources, heldout_seed=6202, arm="SHARED_ONLY"
    )
    with pytest.raises(DiagnosticContractError, match="G1_ALREADY_WRAPPED"):
        install_g1_synthetic_diagnostic(
            model, source_corrections=sources, heldout_seed=6202, arm="FULL"
        )
    with pytest.raises(DiagnosticContractError, match="G1_HANDLE_INACTIVE"):
        remove_g1_synthetic_diagnostic(G1Installation(
            model, original, h.wrapper, h.source_corrections, h.parent_parameters_before,
            h.parent_buffers_before, h.original_layer_mixer_ids, is_installed=False,
        ))
    remove_g1_synthetic_diagnostic(h)
    assert model.mamba.layers[22].mixer is original
    with pytest.raises(DiagnosticContractError, match="G1_HANDLE_INACTIVE"):
        remove_g1_synthetic_diagnostic(h)


def test_g1_context_rolls_back_on_exception_and_mask_reentrancy(fixture):
    model, sources = fixture
    original = model.mamba.layers[22].mixer
    mask = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="synthetic_abort"):
        with installed_g1_synthetic_diagnostic(
            model, source_corrections=sources, heldout_seed=6203,
            arm="RESIDUAL_NORMMATCH",
        ):
            with g1_companion_active_mask(model, mask):
                with pytest.raises(DiagnosticContractError, match="ACTIVE_MASK_REENTRANT"):
                    with g1_companion_active_mask(model, mask):
                        pass
            raise ValueError("synthetic_abort")
    assert model.mamba.layers[22].mixer is original


def test_g1_fail_closed_on_parent_state_and_source_mutations(fixture):
    model, sources = fixture
    model.mamba.config.use_cache = True
    with pytest.raises(DiagnosticContractError, match="G1_CACHE_NOT_DISABLED"):
        install_g1_synthetic_diagnostic(model, source_corrections=sources,
                                        heldout_seed=6201, arm="FULL")
    model.mamba.config.use_cache = False
    model.mamba.layers[22].mixer.scale.requires_grad_(True)
    with pytest.raises(DiagnosticContractError, match="G1_PARENT_NOT_FROZEN"):
        install_g1_synthetic_diagnostic(model, source_corrections=sources,
                                        heldout_seed=6201, arm="FULL")
    model.mamba.layers[22].mixer.scale.requires_grad_(False)
    with torch.no_grad():
        sources[6201].A_theta.weight[0, 0] += 0.1
    with pytest.raises(DiagnosticContractError, match="SOURCE_A_IDENTITY"):
        install_g1_synthetic_diagnostic(model, source_corrections=sources,
                                        heldout_seed=6201, arm="FULL")


def test_g1_parent_buffer_mutation_detected_after_restoration(fixture):
    model, sources = fixture
    native = model.mamba.layers[22].mixer
    h = install_g1_synthetic_diagnostic(model, source_corrections=sources,
                                        heldout_seed=6201, arm="ZERO")
    with torch.no_grad():
        native.marker[0] += 1.0
    with pytest.raises(DiagnosticContractError, match="G1_RESTORE_PARENT_BUFFER_MUTATION"):
        remove_g1_synthetic_diagnostic(h)
    assert model.mamba.layers[22].mixer is native


def test_g1_source_mutation_detected_after_restoration(fixture):
    model, sources = fixture
    native = model.mamba.layers[22].mixer
    h = install_g1_synthetic_diagnostic(model, source_corrections=sources,
                                        heldout_seed=6201, arm="FULL")
    with torch.no_grad():
        sources[6203].B_theta.weight[0, 0] += 0.25
    with pytest.raises(DiagnosticContractError, match="SOURCE_B_IDENTITY"):
        remove_g1_synthetic_diagnostic(h)
    assert model.mamba.layers[22].mixer is native
