"""CPU synthetic numerical equivalence for frozen Gen5 shared-bypass companion.

This test imports the *real repository* legacy recurrence/streaming path, but
uses a synthetic mixer and synthetic rank-2 source corrections throughout.
It never loads pretrained parameters, checkpoint files, datasets, tokenizers,
or performs scientific training/evaluation. This is not runtime evidence from
Mamba-130M and does not authorize any task forward or scientific execution.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contramamba import gen5_shared_bypass_functional_core_diagnostic as diagnostic
from contramamba.gen5_phase2_state_update_ownership import (
    CorrectionShape,
    Phase2Layer22MixerWrapper,
    accelerated_correction_mixer_contribution,
    reference_correction_mixer_contribution,
    validate_runtime_mixer_dimensions,
)


class SyntheticLayer22Mixer(nn.Module):
    """Frozen, model-free tensor fixture exposing the genuine backend API."""

    hidden_size = 768
    intermediate_size = 1536
    ssm_state_size = 16
    time_step_rank = 4
    layer_idx = 22

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(4905)
        self.in_proj = nn.Linear(768, 3072)
        self.conv1d = nn.Conv1d(1536, 1536, kernel_size=1, groups=1536)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(1536, 4 + 16 + 16, bias=False)
        self.dt_proj = nn.Linear(4, 1536)
        self.A_log = nn.Parameter(torch.zeros((1536, 16)), requires_grad=False)
        self.out_proj = nn.Linear(1536, 768)
        self.forward_calls = 0
        self.forward_masks: list[torch.Tensor | None] = []
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name == "A_log":
                    continue
                if name == "dt_proj.bias":
                    param.fill_(-0.3)
                elif param.ndim == 1:
                    param.zero_()
                else:
                    param.normal_(mean=0, std=0.022, generator=generator)
        self.requires_grad_(False)
        self.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ) -> torch.Tensor:
        assert cache_params is None and cache_position is None
        self.forward_calls += 1
        self.forward_masks.append(attention_mask)
        # Deliberately not a pretrained native Mamba forward. The correction
        # recurrence, unlike this synthetic native baseline, IS the real one.
        return hidden_states * 0.375


class SyntheticC0Correction(nn.Module):
    """Synthetic frozen rank-2 factors compatible with both real backends."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.arm = "G5-C0"
        self.shape = CorrectionShape()
        self.A_theta = nn.Linear(768, 2, bias=False)
        self.B_theta = nn.Linear(2, 24576, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.A_theta.weight.normal_(0, 0.032, generator=generator)
            self.B_theta.weight.normal_(0, 0.035, generator=generator)
        self.requires_grad_(False)
        self.eval()

    def forward(self, mixer_input, *, attention_mask=None, return_preproject=False):
        latent = F.linear(mixer_input, self.A_theta.weight)
        raw = F.linear(latent, self.B_theta.weight)
        if attention_mask is not None:
            raw = raw * attention_mask.to(raw.dtype).unsqueeze(-1)
        if return_preproject:
            return raw, raw
        return raw


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def synthetic_case():
    torch.set_num_threads(2)
    mixer = SyntheticLayer22Mixer()
    sources = {seed: SyntheticC0Correction(seed) for seed in diagnostic.SEEDS}
    source_sha = {
        seed: {
            "A_theta.weight": tensor_sha256(source.A_theta.weight),
            "B_theta.weight": tensor_sha256(source.B_theta.weight),
        }
        for seed, source in sources.items()
    }
    x = torch.randn((1, 3, 768), generator=torch.Generator().manual_seed(7)) * 0.6
    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    validate_runtime_mixer_dimensions(mixer)
    return mixer, sources, source_sha, x, mask


@pytest.fixture(autouse=True)
def no_gradient_context():
    with torch.no_grad():
        yield


def assert_near(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=8e-4, atol=1e-5)


def test_existing_reference_equals_existing_streaming_on_synthetic_mixer(synthetic_case):
    mixer, sources, _, x, mask = synthetic_case
    for seed in diagnostic.SEEDS:
        ref = reference_correction_mixer_contribution(
            mixer, x, sources[seed], attention_mask=mask,
        )
        accelerated = accelerated_correction_mixer_contribution(
            mixer, x, sources[seed], attention_mask=mask,
            checkpoint_recompute=False,
        )
        assert tuple(ref.shape) == (1, 3, 768)
        assert_near(ref, accelerated)
        assert torch.isfinite(ref).all()
        assert torch.linalg.vector_norm(ref).item() > 1e-8


def test_companion_full_matches_frozen_phase2_wrapper_all_seeds(synthetic_case, monkeypatch):
    mixer, sources, source_sha, x, mask = synthetic_case
    monkeypatch.setattr(diagnostic, "EXPECTED_SOURCE_TENSOR_SHA256", source_sha)
    diagnostic.validate_frozen_sources(sources)
    for heldout in diagnostic.SEEDS:
        original = Phase2Layer22MixerWrapper(
            native_mixer=mixer, correction=sources[heldout],
        ).eval()
        with original.active_mask(mask):
            expected = original(x)
        companion = diagnostic.FrozenSharedBypassDiagnosticMixer(
            native_mixer=mixer, source_corrections=sources,
            heldout_seed=heldout, arm="FULL",
        ).eval()
        with companion.active_mask(mask):
            actual = companion(x)
        assert_near(actual, expected)
        assert mixer.forward_masks[-1] is None
        assert companion._phase2_active_mask is None
    # CPU read-only path must not alter source tensor bytes.
    for seed, source in sources.items():
        assert tensor_sha256(source.A_theta.weight) == source_sha[seed]["A_theta.weight"]
        assert tensor_sha256(source.B_theta.weight) == source_sha[seed]["B_theta.weight"]


def test_all_six_arms_reconstruct_full_and_normmatch_in_real_streaming(synthetic_case, monkeypatch):
    mixer, sources, source_sha, x, mask = synthetic_case
    monkeypatch.setattr(diagnostic, "EXPECTED_SOURCE_TENSOR_SHA256", source_sha)
    for heldout in diagnostic.SEEDS:
        outputs: dict[str, torch.Tensor] = {}
        for arm in diagnostic.ARMS:
            companion = diagnostic.FrozenSharedBypassDiagnosticMixer(
                native_mixer=mixer, source_corrections=sources,
                heldout_seed=heldout, arm=arm,
            ).eval()
            with companion.active_mask(mask):
                outputs[arm] = companion(x)
        baseline = outputs["ZERO"]
        assert_near(baseline, x * 0.375)
        assert_near(
            outputs["FULL"],
            outputs["SHARED_ONLY"] + outputs["RESIDUAL_ONLY"] - baseline,
        )
        for scaled_arm, source_arm in (
            ("SHARED_NORMMATCH", "SHARED_ONLY"),
            ("RESIDUAL_NORMMATCH", "RESIDUAL_ONLY"),
        ):
            f = diagnostic.FROZEN_ARM_COEFFICIENTS[heldout][scaled_arm]
            g = diagnostic.FROZEN_ARM_COEFFICIENTS[heldout][source_arm]
            first = next(i for i, value in enumerate(g) if value != 0.0)
            scale = f[first] / g[first]
            assert scale > 1.0
            assert_near(
                outputs[scaled_arm],
                baseline + scale * (outputs[source_arm] - baseline),
            )
        assert torch.linalg.vector_norm(outputs["FULL"] - baseline).item() > 1e-7


def test_all_zero_active_mask_suppresses_correction_not_native(synthetic_case, monkeypatch):
    mixer, sources, source_sha, x, _ = synthetic_case
    monkeypatch.setattr(diagnostic, "EXPECTED_SOURCE_TENSOR_SHA256", source_sha)
    zero_mask = torch.zeros((1, 3), dtype=torch.long)
    companion = diagnostic.FrozenSharedBypassDiagnosticMixer(
        native_mixer=mixer, source_corrections=sources,
        heldout_seed=6201, arm="SHARED_NORMMATCH",
    ).eval()
    with companion.active_mask(zero_mask):
        y = companion(x)
    assert_near(y, x * 0.375)
