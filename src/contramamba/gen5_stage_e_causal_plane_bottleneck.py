"""Gen5 Stage E fixed-plane causal bottleneck correction.

Implementation authority:
    33c2c10b8dd14ad4fea826802f3b9163664d2148

This module implements the frozen prospective Stage E parameterization only.
Scientific execution is not authorized by importing this module.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from contramamba.gen5_phase2_state_update_ownership import (
    BASIS_ATOL,
    FROZEN_SHAPE,
    TARGET_LAYER,
    CorrectionShape,
    freeze_parent_parameters,
    parent_parameter_fingerprint,
    validate_basis_geometry,
    validate_runtime_mixer_dimensions,
)


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "33c2c10b8dd14ad4fea826802f3b9163664d2148"
)
ARMS = ("E-R22", "E-C22")
EXPECTED_TRAINABLE_NUMEL = 1540
FIXED_PLANE_RESIDUAL_ATOL = 5e-6


class StageEError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StageEError(message)


def validate_arm(arm: str) -> str:
    require(arm in ARMS, f"UNSUPPORTED_STAGE_E_ARM:{arm}")
    return arm


class FixedPlaneStateWriteCorrection(nn.Module):
    """Rank-2 WRITE22 correction with a frozen output plane Q.

    The realized output matrix is B_eff = Q M.  Only A and M are trainable.
    """

    def __init__(
        self,
        *,
        arm: str,
        r22: torch.Tensor,
        c22: torch.Tensor,
        seed: int,
        shape: CorrectionShape = FROZEN_SHAPE,
        strict_frozen_dimensions: bool = True,
    ) -> None:
        super().__init__()
        self.arm = validate_arm(arm)
        self.shape = shape

        if strict_frozen_dimensions:
            require(shape == FROZEN_SHAPE, "NONFROZEN_STAGE_E_SHAPE")
            require(
                shape.rank * shape.hidden_size + shape.rank * shape.rank
                == EXPECTED_TRAINABLE_NUMEL,
                "STAGE_E_TRAINABLE_NUMEL_CONTRACT",
            )

        validate_basis_geometry(
            r22,
            c22,
            state_width=shape.state_width,
            rank=shape.rank,
            atol=BASIS_ATOL if strict_frozen_dimensions else 1e-8,
        )

        self.A_theta = nn.Linear(shape.hidden_size, shape.rank, bias=False)
        self.M_theta = nn.Linear(shape.rank, shape.rank, bias=False)

        self.register_buffer("R22", r22.detach().clone(), persistent=True)
        self.register_buffer("C22", c22.detach().clone(), persistent=True)

        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        require(type(seed) is int, "SEED_MUST_BE_INT")
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        a_cpu = torch.empty(
            tuple(self.A_theta.weight.shape),
            dtype=torch.float32,
            device="cpu",
        )
        nn.init.kaiming_uniform_(a_cpu, a=math.sqrt(5), generator=g)

        m_cpu = torch.zeros(
            tuple(self.M_theta.weight.shape),
            dtype=torch.float32,
            device="cpu",
        )

        with torch.no_grad():
            self.A_theta.weight.copy_(
                a_cpu.to(
                    device=self.A_theta.weight.device,
                    dtype=self.A_theta.weight.dtype,
                )
            )
            self.M_theta.weight.copy_(
                m_cpu.to(
                    device=self.M_theta.weight.device,
                    dtype=self.M_theta.weight.dtype,
                )
            )

    def selected_basis(self) -> torch.Tensor:
        return self.R22 if self.arm == "E-R22" else self.C22

    def other_basis(self) -> torch.Tensor:
        return self.C22 if self.arm == "E-R22" else self.R22

    def effective_output_matrix(
        self,
        m_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = self.M_theta.weight if m_weight is None else m_weight
        require(
            tuple(m.shape) == (self.shape.rank, self.shape.rank),
            "M_WEIGHT_SHAPE",
        )
        q = self.selected_basis().to(device=m.device, dtype=m.dtype)
        return q @ m

    def forward(
        self,
        mixer_input: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_preproject: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        require(mixer_input.ndim == 3, "MIXER_INPUT_RANK")
        require(
            mixer_input.shape[-1] == self.shape.hidden_size,
            "MIXER_INPUT_WIDTH",
        )

        latent = self.A_theta(mixer_input)
        mixed = self.M_theta(latent)
        q = self.selected_basis().to(
            device=mixed.device,
            dtype=mixed.dtype,
        )
        effective = F.linear(mixed, q, bias=None)

        if attention_mask is not None:
            require(
                tuple(attention_mask.shape) == tuple(mixer_input.shape[:2]),
                "ATTENTION_MASK_SHAPE",
            )
            effective = (
                effective
                * attention_mask.to(effective.dtype).unsqueeze(-1)
            )

        if return_preproject:
            # Stage E has no unconstrained 24576-D preprojection.  Returning
            # the realized write in both slots preserves the frozen reference
            # contribution helper's diagnostic interface without introducing
            # an unauthorized free-B object.
            return effective, effective
        return effective


def fixed_plane_geometry(
    correction: FixedPlaneStateWriteCorrection,
) -> dict[str, float | int]:
    q = correction.selected_basis().detach().cpu().to(torch.float64)
    q_other = correction.other_basis().detach().cpu().to(torch.float64)
    m = correction.M_theta.weight.detach().cpu().to(torch.float64)
    a = correction.A_theta.weight.detach().cpu().to(torch.float64)

    b_eff = q @ m
    projected = q @ (q.T @ b_eff)
    residual = b_eff - projected

    residual_fro = float(torch.linalg.matrix_norm(residual).item())
    residual_max = float(torch.max(torch.abs(residual)).item())
    b_norm = float(torch.linalg.matrix_norm(b_eff).item())
    residual_rel = residual_fro / b_norm if b_norm > 0.0 else 0.0
    cross_max = float(torch.max(torch.abs(q_other.T @ b_eff)).item())

    operator_small = m @ a
    operator_rank = int(torch.linalg.matrix_rank(operator_small).item())
    b_rank = int(torch.linalg.matrix_rank(b_eff).item())

    return {
        "effective_output_rank": b_rank,
        "effective_operator_rank": operator_rank,
        "effective_output_frobenius_norm": b_norm,
        "fixed_plane_residual_frobenius": residual_fro,
        "fixed_plane_residual_max_abs": residual_max,
        "fixed_plane_residual_relative": residual_rel,
        "cross_plane_max_abs": cross_max,
    }


def _streaming_fixed_plane_impl(
    native_mixer: nn.Module,
    correction: FixedPlaneStateWriteCorrection,
    mixer_input: torch.Tensor,
    a_weight: torch.Tensor,
    m_weight: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Memory-bounded correction contribution for B_eff = Q M."""

    shape = correction.shape
    require(mixer_input.ndim == 3, "MIXER_INPUT_RANK")
    batch, seq_len, hidden = mixer_input.shape
    require(hidden == shape.hidden_size, "MIXER_INPUT_WIDTH")
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "ATTENTION_MASK_SHAPE",
    )

    projected = native_mixer.in_proj(mixer_input).transpose(1, 2)
    hidden_states, gate = projected.chunk(2, dim=1)
    active = attention_mask.to(hidden_states.dtype)
    hidden_states = hidden_states * active.unsqueeze(1)

    conv_hidden = native_mixer.act(
        native_mixer.conv1d(hidden_states)[..., :seq_len]
    )
    conv_hidden = conv_hidden * active.unsqueeze(1)

    ssm_parameters = native_mixer.x_proj(conv_hidden.transpose(1, 2))
    time_step, _native_b, c_readout = torch.split(
        ssm_parameters,
        [
            int(native_mixer.time_step_rank),
            int(native_mixer.ssm_state_size),
            int(native_mixer.ssm_state_size),
        ],
        dim=-1,
    )

    discrete_time_step = F.softplus(
        F.linear(
            time_step,
            native_mixer.dt_proj.weight,
            native_mixer.dt_proj.bias,
        )
    ).transpose(1, 2)
    a_continuous = -torch.exp(native_mixer.A_log.float())

    latent = F.linear(mixer_input, a_weight, bias=None)
    q = correction.selected_basis().to(
        device=m_weight.device,
        dtype=m_weight.dtype,
    )
    b_effective = q @ m_weight

    state = torch.zeros(
        (batch, shape.intermediate_size, shape.state_size),
        device=mixer_input.device,
        dtype=latent.dtype,
    )
    outputs: list[torch.Tensor] = []

    for token_index in range(seq_len):
        discrete_a_t = torch.exp(
            a_continuous[None, :, :]
            * discrete_time_step[:, :, token_index, None].float()
        ).to(dtype=latent.dtype)

        write_t = F.linear(
            latent[:, token_index, :],
            b_effective,
            bias=None,
        ).reshape(
            batch,
            shape.intermediate_size,
            shape.state_size,
        )
        write_t = write_t * attention_mask[:, token_index].to(
            write_t.dtype
        )[:, None, None]

        state = discrete_a_t * state + write_t

        read_t = torch.sum(
            state.to(c_readout.dtype)
            * c_readout[:, token_index, None, :],
            dim=-1,
        )
        scan_t = read_t * native_mixer.act(gate[:, :, token_index])
        outputs.append(
            F.linear(
                scan_t,
                native_mixer.out_proj.weight,
                bias=None,
            )
        )

    return torch.stack(outputs, dim=1)


def accelerated_fixed_plane_mixer_contribution(
    native_mixer: nn.Module,
    mixer_input: torch.Tensor,
    correction: FixedPlaneStateWriteCorrection,
    *,
    attention_mask: torch.Tensor,
    checkpoint_recompute: bool = True,
) -> torch.Tensor:
    """Checkpointed streaming backend with explicit A/M gradient inputs."""

    require(attention_mask is not None, "STAGE_E_ACTIVE_MASK_REQUIRED")
    args = (
        mixer_input,
        correction.A_theta.weight,
        correction.M_theta.weight,
        attention_mask,
    )

    def run(
        input_tensor: torch.Tensor,
        a_weight: torch.Tensor,
        m_weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return _streaming_fixed_plane_impl(
            native_mixer,
            correction,
            input_tensor,
            a_weight,
            m_weight,
            mask,
        )

    if checkpoint_recompute and torch.is_grad_enabled():
        return checkpoint(
            run,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )
    return run(*args)


class StageELayer22MixerWrapper(nn.Module):
    """Exact native layer-22 mixer plus a fixed-plane correction branch."""

    def __init__(
        self,
        *,
        native_mixer: nn.Module,
        correction: FixedPlaneStateWriteCorrection,
        strict_frozen_dimensions: bool = True,
    ) -> None:
        super().__init__()
        if strict_frozen_dimensions:
            validate_runtime_mixer_dimensions(native_mixer)

        self.native_mixer = native_mixer
        self.correction = correction
        self.layer_idx = int(getattr(native_mixer, "layer_idx", TARGET_LAYER))
        self._stage_e_active_mask: torch.Tensor | None = None

        require(
            self.layer_idx == TARGET_LAYER or not strict_frozen_dimensions,
            f"TARGET_LAYER_MISMATCH:{self.layer_idx}",
        )

    @contextmanager
    def active_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> Iterator[None]:
        require(
            self._stage_e_active_mask is None,
            "STAGE_E_ACTIVE_MASK_REENTRANT",
        )
        require(attention_mask.ndim == 2, "STAGE_E_ACTIVE_MASK_RANK")
        self._stage_e_active_mask = attention_mask
        try:
            yield
        finally:
            self._stage_e_active_mask = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        require(
            cache_params is None,
            "STAGE_E_REFERENCE_BACKEND_CACHE_UNSUPPORTED",
        )
        require(
            cache_position is None,
            "STAGE_E_REFERENCE_BACKEND_CACHE_POSITION_UNSUPPORTED",
        )

        effective_mask = (
            attention_mask
            if attention_mask is not None
            else self._stage_e_active_mask
        )
        require(
            effective_mask is not None,
            "STAGE_E_ACTIVE_MASK_REQUIRED",
        )
        require(
            tuple(effective_mask.shape)
            == tuple(hidden_states.shape[:2]),
            "STAGE_E_ACTIVE_MASK_SHAPE",
        )

        native_output = self.native_mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        correction_output = accelerated_fixed_plane_mixer_contribution(
            self.native_mixer,
            hidden_states,
            self.correction,
            attention_mask=effective_mask,
            checkpoint_recompute=True,
        )
        return native_output + correction_output


def stage_e_parameter_audit(model: nn.Module) -> dict[str, Any]:
    trainable = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    names = [name for name, _ in trainable]
    numel = int(sum(p.numel() for _, p in trainable))
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_numel": numel,
        "trainable_names": names,
        "all_trainable_are_correction": all(
            ".correction." in name for name in names
        ),
    }


def stage_e_optimizer_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    audit = stage_e_parameter_audit(model)
    require(audit["trainable_tensor_count"] == 2, "OPTIMIZER_TENSOR_COUNT")
    require(
        audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL,
        "OPTIMIZER_TRAINABLE_NUMEL",
    )
    require(
        audit["all_trainable_are_correction"],
        "OPTIMIZER_OWNERSHIP",
    )

    parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and ".correction." in name
    ]
    require(len(parameters) == 2, "OPTIMIZER_PARAMETER_LIST")
    return parameters


def install_stage_e_layer22_wrapper(
    model: nn.Module,
    *,
    arm: str,
    r22: torch.Tensor,
    c22: torch.Tensor,
    seed: int,
    shape: CorrectionShape = FROZEN_SHAPE,
    strict_frozen_dimensions: bool = True,
) -> StageELayer22MixerWrapper:
    validate_arm(arm)
    require(hasattr(model, "mamba"), "MODEL_MAMBA_MISSING")
    require(hasattr(model.mamba, "layers"), "MAMBA_LAYERS_MISSING")
    require(len(model.mamba.layers) > TARGET_LAYER, "LAYER22_MISSING")

    target_block = model.mamba.layers[TARGET_LAYER]
    native_mixer = target_block.mixer
    require(
        not isinstance(native_mixer, StageELayer22MixerWrapper),
        "LAYER22_ALREADY_WRAPPED",
    )

    if strict_frozen_dimensions:
        validate_runtime_mixer_dimensions(native_mixer)

    before = parent_parameter_fingerprint(model)
    freeze_parent_parameters(model)

    native_weight = native_mixer.in_proj.weight
    correction = FixedPlaneStateWriteCorrection(
        arm=arm,
        r22=r22,
        c22=c22,
        seed=seed,
        shape=shape,
        strict_frozen_dimensions=strict_frozen_dimensions,
    )
    correction.to(
        device=native_weight.device,
        dtype=native_weight.dtype,
    )

    # Keep frozen scientific bases in their authenticated precision.
    correction.R22 = r22.detach().clone().to(
        device=native_weight.device,
        dtype=torch.float64,
    )
    correction.C22 = c22.detach().clone().to(
        device=native_weight.device,
        dtype=torch.float64,
    )

    wrapper = StageELayer22MixerWrapper(
        native_mixer=native_mixer,
        correction=correction,
        strict_frozen_dimensions=strict_frozen_dimensions,
    )
    target_block.mixer = wrapper

    after = parent_parameter_fingerprint(model)
    require(
        before == after,
        "PARENT_PARAMETER_FINGERPRINT_CHANGED_ON_INSTALL",
    )

    audit = stage_e_parameter_audit(model)
    require(audit["trainable_tensor_count"] == 2, "TRAINABLE_TENSOR_COUNT")
    if strict_frozen_dimensions:
        require(
            audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL,
            "TRAINABLE_NUMEL",
        )
    require(
        audit["all_trainable_are_correction"],
        "NONCORRECTION_TRAINABLE_PARAMETER",
    )
    return wrapper


def get_stage_e_layer22_wrapper(
    model: nn.Module,
) -> StageELayer22MixerWrapper:
    require(hasattr(model, "mamba"), "MODEL_MAMBA_MISSING")
    wrapper = model.mamba.layers[TARGET_LAYER].mixer
    require(
        isinstance(wrapper, StageELayer22MixerWrapper),
        "LAYER22_NOT_STAGE_E_WRAPPED",
    )
    return wrapper


@contextmanager
def stage_e_active_mask(
    model: nn.Module,
    attention_mask: torch.Tensor,
) -> Iterator[None]:
    wrapper = get_stage_e_layer22_wrapper(model)
    with wrapper.active_mask(attention_mask):
        yield


def remove_stage_e_layer22_wrapper(model: nn.Module) -> nn.Module:
    wrapper = get_stage_e_layer22_wrapper(model)
    native = wrapper.native_mixer
    model.mamba.layers[TARGET_LAYER].mixer = native
    return native
