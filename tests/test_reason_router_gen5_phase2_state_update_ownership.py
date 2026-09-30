from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contramamba.gen5_phase2_state_update_ownership import (
    ARMS,
    accelerated_correction_mixer_contribution,
    CorrectionShape,
    Phase2Layer22MixerWrapper,
    StateWriteCorrection,
    correction_parameter_audit,
    explicit_total_recurrence,
    parent_parameter_fingerprint,
    phase2_active_mask,
    phase2_final_three_way_ce,
    reference_correction_mixer_contribution,
    reference_state_stack_bytes,
    streaming_state_bytes,
    reference_correction_recurrence,
    remove_phase2_layer22_wrapper,
    validate_arm,
)


TEST_SHAPE = CorrectionShape(
    hidden_size=4,
    intermediate_size=3,
    state_size=2,
    rank=2,
)


def orthogonal_bases(state_width: int):
    eye = torch.eye(state_width, dtype=torch.float64)
    return eye[:, :2].clone(), eye[:, 2:4].clone()


class TinyMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = TEST_SHAPE.hidden_size
        self.intermediate_size = TEST_SHAPE.intermediate_size
        self.ssm_state_size = TEST_SHAPE.state_size
        self.time_step_rank = 2
        self.layer_idx = 22
        self.in_proj = nn.Linear(4, 6, bias=False)
        self.conv1d = nn.Conv1d(
            3, 3, kernel_size=2, padding=1, groups=3, bias=True
        )
        self.x_proj = nn.Linear(3, 2 + 2 * 2, bias=False)
        self.dt_proj = nn.Linear(2, 3, bias=True)
        self.A_log = nn.Parameter(torch.zeros(3, 2))
        self.D = nn.Parameter(torch.ones(3))
        self.out_proj = nn.Linear(3, 4, bias=True)
        self.act = F.silu

    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        # A bounded native path sufficient for wrapper zero-equivalence tests.
        x = self.in_proj(hidden_states)[..., :3]
        return self.out_proj(torch.tanh(x))


class TinyBlock(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [TinyBlock(nn.Identity()) for _ in range(23)]
        layers[22] = TinyBlock(TinyMixer())
        self.mamba = SimpleNamespace(layers=layers)
        # Register target mixer parameters in a normal Module container too.
        self.target = layers[22].mixer


def make_correction(arm: str, seed: int = 5201):
    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    return StateWriteCorrection(
        arm=arm,
        r22=r,
        c22=c,
        seed=seed,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )


def test_arm_validation():
    for arm in ARMS:
        assert validate_arm(arm) == arm
    with pytest.raises(Exception):
        validate_arm("G5-X")


def test_initialization_same_seed_and_zero_output():
    x = torch.randn(2, 5, 4)
    a = make_correction("G5-C0", 5201)
    b = make_correction("G5-M1", 5201)
    assert torch.equal(a.A_theta.weight, b.A_theta.weight)
    assert torch.equal(a.B_theta.weight, b.B_theta.weight)
    assert torch.count_nonzero(a.B_theta.weight) == 0
    assert torch.count_nonzero(a(x)) == 0


def test_different_seed_changes_a():
    a = make_correction("G5-C0", 5201)
    b = make_correction("G5-C0", 5202)
    assert not torch.equal(a.A_theta.weight, b.A_theta.weight)


def test_projector_and_autograd():
    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    raw = torch.randn(7, TEST_SHAPE.state_width, requires_grad=True)

    m1 = make_correction("G5-M1")
    c1 = make_correction("G5-C1")
    c0 = make_correction("G5-C0")

    m1_out = m1.project(raw)
    c1_out = c1.project(raw)
    c0_out = c0.project(raw)

    assert torch.max(torch.abs(m1_out.double() @ r)).item() < 1e-5
    assert torch.max(torch.abs(c1_out.double() @ c)).item() < 1e-5
    assert torch.equal(c0_out, raw)

    (m1_out.square().sum() + c1_out.square().sum()).backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert m1.R22.grad is None
    assert m1.C22.grad is None


def test_recurrence_superposition_and_persistence():
    torch.manual_seed(1)
    b, i, l, n = 2, 3, 4, 2
    discrete_a = torch.sigmoid(torch.randn(b, i, l, n))
    native_write = torch.randn(b, l, i, n)
    correction_write = torch.randn(b, l, i, n)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)

    native, correction, direct = explicit_total_recurrence(
        discrete_a,
        native_write,
        correction_write,
        active_mask=mask,
    )
    assert torch.allclose(native + correction, direct, atol=1e-6, rtol=1e-6)

    zero = reference_correction_recurrence(
        discrete_a,
        torch.zeros_like(correction_write),
        active_mask=mask,
    )
    assert torch.count_nonzero(zero) == 0

    # No new write at t=2 for row 0, but previous correction persists through decay.
    assert torch.linalg.vector_norm(correction[0, 2]).item() > 0


def test_wrapper_zero_equivalence_and_backward_isolation():
    mixer = TinyMixer()
    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    corr = StateWriteCorrection(
        arm="G5-M1",
        r22=r,
        c22=c,
        seed=5201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )
    for p in mixer.parameters():
        p.requires_grad_(False)

    wrapper = Phase2Layer22MixerWrapper(
        native_mixer=mixer,
        correction=corr,
        strict_frozen_dimensions=False,
    )
    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        native = mixer(x, attention_mask=mask)
        wrapped = wrapper(x, attention_mask=mask)
    assert torch.equal(native, wrapped)

    # Make correction nonzero and verify only correction receives gradients.
    with torch.no_grad():
        corr.B_theta.weight.normal_(mean=0.0, std=0.01)

    out = wrapper(x, attention_mask=mask)
    loss = out.square().mean()
    loss.backward()
    assert corr.A_theta.weight.grad is not None
    assert corr.B_theta.weight.grad is not None
    assert torch.isfinite(corr.A_theta.weight.grad).all()
    assert torch.isfinite(corr.B_theta.weight.grad).all()
    assert all(p.grad is None for p in mixer.parameters())


def test_final_ce_is_plain_cross_entropy():
    logits = torch.tensor(
        [[1.0, 0.5, -0.5], [0.1, -0.2, 0.8]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 2])
    observed = phase2_final_three_way_ce(logits, labels)
    expected = F.cross_entropy(logits, labels)
    assert torch.equal(observed, expected)


class FrozenShapeMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.intermediate_size = 1536
        self.ssm_state_size = 16
        self.time_step_rank = 48
        self.layer_idx = 22
        self.in_proj = nn.Linear(768, 3072, bias=False)
        self.conv1d = nn.Conv1d(
            1536, 1536, kernel_size=4, padding=3, groups=1536, bias=True
        )
        self.x_proj = nn.Linear(1536, 48 + 32, bias=False)
        self.dt_proj = nn.Linear(48, 1536, bias=True)
        self.A_log = nn.Parameter(torch.zeros(1536, 16))
        self.D = nn.Parameter(torch.ones(1536))
        self.out_proj = nn.Linear(1536, 768, bias=False)
        self.act = F.silu

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return hidden_states


class RegisteredBlock(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


class RegisteredMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [RegisteredBlock(nn.Identity()) for _ in range(23)]
        )
        self.layers[22] = RegisteredBlock(FrozenShapeMixer())


class RegisteredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = RegisteredMamba()


def test_production_install_scope_mask_bridge_and_remove():
    from contramamba.gen5_phase2_state_update_ownership import (
        EXPECTED_TRAINABLE_NUMEL,
        Phase2Layer22MixerWrapper,
        install_phase2_layer22_wrapper,
    )

    model = RegisteredModel()
    original = model.mamba.layers[22].mixer
    untouched = [model.mamba.layers[i].mixer for i in range(22)]
    before = parent_parameter_fingerprint(model)

    r22 = torch.zeros(24576, 2, dtype=torch.float64)
    c22 = torch.zeros(24576, 2, dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0

    wrapper = install_phase2_layer22_wrapper(
        model,
        arm="G5-M1",
        r22=r22,
        c22=c22,
        seed=5201,
    )
    assert isinstance(model.mamba.layers[22].mixer, Phase2Layer22MixerWrapper)
    assert wrapper.native_mixer is original
    assert all(model.mamba.layers[i].mixer is untouched[i] for i in range(22))
    assert parent_parameter_fingerprint(model) == before

    audit = correction_parameter_audit(model)
    assert audit["trainable_tensor_count"] == 2
    assert audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL
    assert audit["all_trainable_are_correction"]

    # Historical model does not forward its attention mask to Mamba.
    x = torch.randn(1, 3, 768)
    with pytest.raises(Exception):
        wrapper(x)

    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    with phase2_active_mask(model, mask):
        out = wrapper(x)
    assert torch.equal(out, x)  # zero initialized correction

    restored = remove_phase2_layer22_wrapper(model)
    assert restored is original
    assert model.mamba.layers[22].mixer is original
    assert parent_parameter_fingerprint(model) == before


def test_bases_are_buffers_not_parameters():
    module = make_correction("G5-M1")
    parameter_names = dict(module.named_parameters())
    buffer_names = dict(module.named_buffers())
    assert "R22" not in parameter_names
    assert "C22" not in parameter_names
    assert "R22" in buffer_names
    assert "C22" in buffer_names


def _copy_nonzero_correction(source, target):
    with torch.no_grad():
        target.A_theta.weight.copy_(source.A_theta.weight)
        target.B_theta.weight.copy_(source.B_theta.weight)


def test_accelerated_matches_reference_forward_and_gradients():
    torch.manual_seed(17)
    mixer = TinyMixer()
    for parameter in mixer.parameters():
        parameter.requires_grad_(False)

    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    ref_corr = StateWriteCorrection(
        arm="G5-M1",
        r22=r,
        c22=c,
        seed=5201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )
    acc_corr = StateWriteCorrection(
        arm="G5-M1",
        r22=r,
        c22=c,
        seed=5201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )
    with torch.no_grad():
        ref_corr.B_theta.weight.normal_(0.0, 0.02)
    _copy_nonzero_correction(ref_corr, acc_corr)

    x = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )

    ref = reference_correction_mixer_contribution(
        mixer,
        x,
        ref_corr,
        attention_mask=mask,
    )
    acc = accelerated_correction_mixer_contribution(
        mixer,
        x,
        acc_corr,
        attention_mask=mask,
        checkpoint_recompute=True,
    )
    assert torch.allclose(ref, acc, atol=1e-6, rtol=1e-6)

    ref_loss = ref.square().sum()
    acc_loss = acc.square().sum()
    ref_loss.backward()
    acc_loss.backward()

    assert torch.allclose(
        ref_corr.A_theta.weight.grad,
        acc_corr.A_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        ref_corr.B_theta.weight.grad,
        acc_corr.B_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_streaming_backend_removes_sequence_state_stack():
    batch = 2880
    seq = 128
    reference = reference_state_stack_bytes(batch, seq, dtype_bytes=4)
    streaming = streaming_state_bytes(batch, dtype_bytes=4)
    assert reference == 36_238_786_560
    assert streaming == 283_115_520
    assert reference == streaming * seq
