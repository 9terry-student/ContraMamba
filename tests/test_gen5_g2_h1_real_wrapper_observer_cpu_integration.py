"""CPU-only integration contract: real frozen G2 wrapper + G2-H1 observer.

Uses repository's preexisting synthetic parent/donor fixtures. No pretrained
model, tokenizer, checkpoint, real task data, GPU, training or evaluation.
"""
from __future__ import annotations

import pytest
import torch

from contramamba import gen5_shared_bypass_functional_core_diagnostic as g2
from contramamba.gen5_g2_h1_layer22_output_observer import (
    Layer22OutputObserver,
    ObservationContractError,
    final_logit_rs_contrasts,
)
from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
    g2_companion_active_mask,
    installed_g2_loaded_parent,
)
from test_gen5_shared_bypass_g1_installation_synthetic import (
    FakeParent,
    TinyC0,
    fake_correction,
)


@pytest.fixture
def setup(monkeypatch):
    model = FakeParent()
    sources = {seed: TinyC0(seed) for seed in g2.SEEDS}
    identities = {
        seed: {
            "A_theta.weight": g2._tensor_sha256(src.A_theta.weight),
            "B_theta.weight": g2._tensor_sha256(src.B_theta.weight),
        }
        for seed, src in sources.items()
    }
    monkeypatch.setattr(g2, "EXPECTED_SOURCE_TENSOR_SHA256", identities)
    monkeypatch.setattr(g2, "accelerated_correction_mixer_contribution", fake_correction)
    return model, sources


@pytest.fixture
def rows():
    x = torch.zeros((4, 4, 768), dtype=torch.float32)
    x[0, :, 0] = torch.tensor([2., 3., 4., 999.])
    x[1, :, 0] = torch.tensor([-1., 5., 6., 999.])
    x[2, :, 0] = torch.tensor([7., 1., 2., 999.])
    x[3, :, 0] = torch.tensor([4., 2., 9., 999.])
    x[:, :, 1] = 0.25
    valid = torch.tensor([
        [1, 1, 1, 0], [1, 1, 1, 0],
        [1, 1, 1, 0], [1, 1, 1, 0],
    ], dtype=torch.long)
    claim = torch.tensor([[1, 0, 0, 0]] * 4, dtype=torch.long)
    evidence = torch.tensor([[0, 0, 1, 0]] * 4, dtype=torch.long)
    pairs = ("pair-b", "pair-a", "pair-b", "pair-a")
    cells = ("C6_EXPLICIT_DENIAL", "C0_SHAM", "C0_SHAM", "C6_EXPLICIT_DENIAL")
    return x, valid, claim, evidence, pairs, cells


def observer_for(wrapper, rows):
    _x, valid, claim, evidence, pairs, cells = rows
    return Layer22OutputObserver(
        wrapper, attention_mask=valid, claim_mask=claim,
        evidence_mask=evidence, pair_ids=pairs, cell_ids=cells,
        row_ids=("b6", "a0", "b0", "a6"),
    )


def synthetic_logits(output: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """A fixed arithmetic projection only; these are NOT scientific task logits."""
    pooled = (output * valid.unsqueeze(-1)).sum(dim=1)
    return torch.stack((pooled[:, 0], pooled[:, 1], -pooled[:, 0]), dim=-1)


@pytest.mark.parametrize("arm", ("ZERO", "FULL", "SHARED_ONLY"))
def test_real_g2_wrapper_observer_equivalence_and_correction(setup, rows, arm):
    model, sources = setup
    x, valid, _claim, _evidence, pairs, cells = rows
    native = model.mamba.layers[22].mixer
    before_parameters = [p.detach().clone() for p in model.parameters()]
    assert not native._forward_hooks

    with installed_g2_loaded_parent(
        model, source_corrections=sources, heldout_seed=6201, arm=arm,
    ) as wrapper:
        assert type(wrapper) is g2.FrozenSharedBypassDiagnosticMixer
        with torch.inference_mode(), g2_companion_active_mask(model, valid):
            original = model.mamba(input_ids=x)
            original_logits = synthetic_logits(original, valid)
        original_native_calls = len(native.masks)

        observation = observer_for(wrapper, rows)
        with torch.inference_mode(), g2_companion_active_mask(model, valid), observation:
            hooked = model.mamba(input_ids=x)
            hooked_logits = synthetic_logits(hooked, valid)
        record = observation.record()

        torch.testing.assert_close(hooked, original, atol=0, rtol=0)
        torch.testing.assert_close(hooked_logits, original_logits, atol=0, rtol=0)
        assert len(native.masks) == original_native_calls + 1
        assert native.masks[-1] is None  # out-of-band mask never reaches native
        torch.testing.assert_close(record.native, native.scale * x, atol=0, rtol=0)
        torch.testing.assert_close(record.wrapped, original, atol=0, rtol=0)

        expected = torch.zeros_like(record.contribution)
        for seed, coeff in zip(g2.SEEDS, wrapper.coefficients):
            if coeff:
                expected = expected + coeff * fake_correction(
                    native, x, sources[seed], attention_mask=valid,
                    checkpoint_recompute=False,
                )
        # wrapped - native rounds in output dtype; compare with tolerance.
        torch.testing.assert_close(record.contribution, expected, atol=2e-6, rtol=0)
        assert record.per_row[0]["contribution"]["valid"]["tokens"] == 3
        assert record.per_row[0]["contribution"]["claim"]["tokens"] == 1
        assert record.per_row[0]["contribution"]["evidence"]["tokens"] == 1
        assert record.per_row[0]["contribution"]["other"]["tokens"] == 1
        assert set(record.pair_differences) == {"pair-a", "pair-b"}
        assert final_logit_rs_contrasts(hooked_logits, pairs, cells) == (
            final_logit_rs_contrasts(original_logits, pairs, cells)
        )
        if arm == "ZERO":
            assert torch.count_nonzero(record.contribution).item() == 0
        else:
            assert torch.count_nonzero(record.contribution).item() > 0
        assert not wrapper._forward_hooks and not native._forward_hooks
        assert wrapper._phase2_active_mask is None
    assert model.mamba.layers[22].mixer is native
    assert all(torch.equal(before, after) for before, after in zip(
        before_parameters, model.parameters()
    ))


def test_real_g2_explicit_mask_is_forwarded_without_rewriting(setup, rows):
    model, sources = setup
    x, valid, *_ = rows
    with installed_g2_loaded_parent(
        model, source_corrections=sources, heldout_seed=6202, arm="FULL",
    ) as wrapper:
        with torch.inference_mode(), observer_for(wrapper, rows) as observation:
            wrapper(x, attention_mask=valid)
        assert wrapper.native_mixer.masks[-1] is valid
        assert observation.record().wrapped.shape == x.shape
        assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks


def test_real_g2_wrong_effective_mask_fails_closed(setup, rows):
    model, sources = setup
    x, valid, *_ = rows
    wrong = valid.clone()
    wrong[0, 2] = 0
    native = model.mamba.layers[22].mixer
    with installed_g2_loaded_parent(
        model, source_corrections=sources, heldout_seed=6203, arm="FULL",
    ) as wrapper:
        observation = observer_for(wrapper, rows)
        with pytest.raises(ObservationContractError,
                           match="OBSERVER_EFFECTIVE_MASK_MISMATCH"):
            with torch.inference_mode(), g2_companion_active_mask(model, wrong), observation:
                model.mamba(input_ids=x)
        assert not wrapper._forward_hooks and not native._forward_hooks
        assert wrapper._phase2_active_mask is None
        with pytest.raises(ObservationContractError, match="NOT_COMPLETE"):
            observation.record()
    assert model.mamba.layers[22].mixer is native


def test_real_g2_correction_exception_cleans_hooks_and_restores(setup, rows, monkeypatch):
    model, sources = setup
    x, valid, *_ = rows
    native = model.mamba.layers[22].mixer

    def fail_after_native(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_CORRECTION_FAILURE")

    monkeypatch.setattr(g2, "accelerated_correction_mixer_contribution",
                        fail_after_native)
    with pytest.raises(RuntimeError, match="SYNTHETIC_CORRECTION_FAILURE"):
        with installed_g2_loaded_parent(
            model, source_corrections=sources, heldout_seed=6201, arm="FULL",
        ) as wrapper:
            with torch.inference_mode(), g2_companion_active_mask(model, valid), observer_for(wrapper, rows):
                model.mamba(input_ids=x)
    assert model.mamba.layers[22].mixer is native
    assert not native._forward_hooks
