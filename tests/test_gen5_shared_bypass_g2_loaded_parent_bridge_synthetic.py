"""G2 standalone diagnostic bridge tests: synthetic CPU parent only.

No pretrained model construction, no checkpoint read, no GPU, no task data.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from contramamba import gen5_shared_bypass_functional_core_diagnostic as companion
from contramamba.gen5_phase2_state_update_ownership import (
    parent_parameter_fingerprint,
    phase2_active_mask,
)
from contramamba.gen5_shared_bypass_g1_installation_diagnostic import (
    _native_buffer_sha256,
)
from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
    DiagnosticContractError,
    g2_companion_active_mask,
    install_g2_loaded_parent,
    installed_g2_loaded_parent,
    remove_g2_loaded_parent,
)
from test_gen5_shared_bypass_g1_installation_synthetic import (
    FakeParent, TinyC0, fake_correction,
)


@pytest.fixture
def fixture(monkeypatch):
    model = FakeParent()
    sources = {seed: TinyC0(seed) for seed in companion.SEEDS}
    hashes = {seed: {
        "A_theta.weight": companion._tensor_sha256(src.A_theta.weight),
        "B_theta.weight": companion._tensor_sha256(src.B_theta.weight),
    } for seed, src in sources.items()}
    monkeypatch.setattr(companion, "EXPECTED_SOURCE_TENSOR_SHA256", hashes)
    monkeypatch.setattr(companion, "accelerated_correction_mixer_contribution", fake_correction)
    return model, sources


def test_g2_injected_model_has_unchanged_parent_and_restoration(fixture):
    model, sources = fixture
    native = model.mamba.layers[22].mixer
    params = parent_parameter_fingerprint(model)
    buffers = _native_buffer_sha256(model)
    ids = tuple(id(x.mixer) for x in model.mamba.layers)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    x = torch.ones(1, 3, 768)
    with installed_g2_loaded_parent(model, source_corrections=sources,
                                    heldout_seed=6201, arm="FULL") as wrapper:
        assert wrapper.native_mixer is native
        assert not wrapper.training and wrapper._phase2_active_mask is None
        with torch.no_grad(), g2_companion_active_mask(model, mask):
            y = model.mamba(input_ids=x)
        assert torch.isfinite(y).all() and y.shape == x.shape
        assert native.masks[-1] is None
        assert wrapper._phase2_active_mask is None
        assert parent_parameter_fingerprint(model) == params
        assert _native_buffer_sha256(model) == buffers
        with pytest.raises(Exception, match="LAYER22_NOT_WRAPPED"):
            with phase2_active_mask(model, mask):
                pass
    assert tuple(id(z.mixer) for z in model.mamba.layers) == ids
    assert model.mamba.layers[22].mixer is native
    assert parent_parameter_fingerprint(model) == params
    assert _native_buffer_sha256(model) == buffers


def test_g2_all_18_arm_installation_restore_without_task_eval(fixture):
    model, sources = fixture
    native = model.mamba.layers[22].mixer
    mask = torch.tensor([[1, 1]], dtype=torch.long)
    x = torch.ones(1, 2, 768)
    for seed in companion.SEEDS:
        for arm in companion.ARMS:
            with torch.no_grad(), installed_g2_loaded_parent(
                model, source_corrections=sources, heldout_seed=seed, arm=arm,
            ) as wrapper:
                with g2_companion_active_mask(model, mask):
                    y = model.mamba(input_ids=x)
                assert torch.isfinite(y).all() and not wrapper.training
            assert model.mamba.layers[22].mixer is native


def test_g2_rejects_duplicate_and_legacy_wrapper(fixture):
    model, sources = fixture
    h = install_g2_loaded_parent(model, source_corrections=sources,
                                 heldout_seed=6202, arm="ZERO")
    with pytest.raises(DiagnosticContractError, match="G2_ALREADY_WRAPPED"):
        install_g2_loaded_parent(model, source_corrections=sources,
                                 heldout_seed=6202, arm="ZERO")
    remove_g2_loaded_parent(h)
    with pytest.raises(DiagnosticContractError, match="G2_HANDLE_INACTIVE"):
        remove_g2_loaded_parent(h)


def test_g2_rejects_training_cache_and_source_device_mismatch(fixture):
    model, sources = fixture
    model.train()
    with pytest.raises(DiagnosticContractError, match="G2_MODEL_NOT_EVAL"):
        install_g2_loaded_parent(model, source_corrections=sources,
                                 heldout_seed=6202, arm="FULL")
    model.eval()
    model.mamba.config.use_cache = True
    with pytest.raises(DiagnosticContractError, match="G1_CACHE_NOT_DISABLED"):
        install_g2_loaded_parent(model, source_corrections=sources,
                                 heldout_seed=6202, arm="FULL")
    model.mamba.config.use_cache = False
    sources[6201].B_theta = nn.Linear(2, 24576, bias=False, device="meta")
    with pytest.raises(DiagnosticContractError,
                       match="G2_SOURCE_6201_MULTIPLE_OR_EMPTY_DEVICES"):
        install_g2_loaded_parent(model, source_corrections=sources,
                                 heldout_seed=6202, arm="FULL")


def test_g2_exception_rollback_and_active_mask_reentrancy(fixture):
    model, sources = fixture
    native = model.mamba.layers[22].mixer
    mask = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="G2_ABORT"):
        with installed_g2_loaded_parent(model, source_corrections=sources,
                                        heldout_seed=6203, arm="SHARED_ONLY"):
            with g2_companion_active_mask(model, mask):
                with pytest.raises(DiagnosticContractError, match="ACTIVE_MASK_REENTRANT"):
                    with g2_companion_active_mask(model, mask):
                        pass
            raise ValueError("G2_ABORT")
    assert model.mamba.layers[22].mixer is native


def test_g2_detects_buffer_tampering_after_restoration(fixture):
    model, sources = fixture
    native = model.mamba.layers[22].mixer
    handle = install_g2_loaded_parent(model, source_corrections=sources,
                                      heldout_seed=6203, arm="RESIDUAL_ONLY")
    with torch.no_grad():
        native.marker[0] += 1.0
    with pytest.raises(DiagnosticContractError, match="G2_RESTORE_PARENT_BUFFER_MUTATION"):
        remove_g2_loaded_parent(handle)
    assert model.mamba.layers[22].mixer is native
