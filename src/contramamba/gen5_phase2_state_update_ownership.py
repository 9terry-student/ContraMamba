"""Gen5 Phase 2 state-update ownership correction.

This module is intentionally narrow. It implements only the frozen Phase 2
WRITE22 correction mechanics authorized at e2f8975d8271c0e95c92b9389dfc4717a221f7df.

Scientific training/evaluation is not authorized by this module.
"""

from __future__ import annotations

import hashlib
import math
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F


PHASE2_IMPLEMENTATION_AUTHORITY_COMMIT = (
    "e2f8975d8271c0e95c92b9389dfc4717a221f7df"
)

TARGET_LAYER = 22
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
STATE_SIZE = 16
STATE_WIDTH = INTERMEDIATE_SIZE * STATE_SIZE
CORRECTION_RANK = 2
EXPECTED_TRAINABLE_NUMEL = 50688
BASIS_ATOL = 1e-10

R22_SHA256 = "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
C22_SHA256 = "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"

BASIS_ROOT = Path(
    "reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1"
)
R22_RELATIVE_PATH = BASIS_ROOT / "r22_basis.f64le"
C22_RELATIVE_PATH = BASIS_ROOT / "c22_basis.f64le"

ARMS = ("G5-C0", "G5-C1", "G5-M1")


class Phase2OwnershipError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Phase2OwnershipError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_arm(arm: str) -> str:
    require(arm in ARMS, f"UNSUPPORTED_PHASE2_ARM:{arm}")
    return arm


def _basis_from_bytes(
    raw: bytes,
    *,
    state_width: int = STATE_WIDTH,
    rank: int = CORRECTION_RANK,
    label: str,
) -> torch.Tensor:
    require(len(raw) == state_width * rank * 8, f"{label}_BYTES")
    values = struct.unpack(f"<{state_width * rank}d", raw)
    matrix = (
        torch.tensor(values, dtype=torch.float64)
        .reshape(rank, state_width)
        .T
        .contiguous()
    )
    require(tuple(matrix.shape) == (state_width, rank), f"{label}_SHAPE")
    require(bool(torch.isfinite(matrix).all().item()), f"{label}_FINITE")
    return matrix


def validate_basis_geometry(
    r22: torch.Tensor,
    c22: torch.Tensor,
    *,
    state_width: int = STATE_WIDTH,
    rank: int = CORRECTION_RANK,
    atol: float = BASIS_ATOL,
) -> dict[str, float]:
    require(tuple(r22.shape) == (state_width, rank), "R22_SHAPE")
    require(tuple(c22.shape) == (state_width, rank), "C22_SHAPE")
    require(r22.dtype == torch.float64, "R22_DTYPE")
    require(c22.dtype == torch.float64, "C22_DTYPE")
    require(bool(torch.isfinite(r22).all().item()), "R22_NONFINITE")
    require(bool(torch.isfinite(c22).all().item()), "C22_NONFINITE")

    eye = torch.eye(rank, dtype=torch.float64, device=r22.device)
    c_eye = eye.to(c22.device)
    r_residual = float(torch.max(torch.abs(r22.T @ r22 - eye)).item())
    c_residual = float(torch.max(torch.abs(c22.T @ c22 - c_eye)).item())
    cross = float(
        torch.max(
            torch.abs(r22.T.to(device=c22.device) @ c22)
        ).item()
    )
    require(r_residual <= atol, f"R22_ORTHONORMALITY:{r_residual}")
    require(c_residual <= atol, f"C22_ORTHONORMALITY:{c_residual}")
    require(cross <= atol, f"R22_C22_CROSS_ORTHOGONALITY:{cross}")
    return {
        "r22_orthonormality_max_abs": r_residual,
        "c22_orthonormality_max_abs": c_residual,
        "r22_c22_cross_max_abs": cross,
    }


def load_frozen_owner_bases(
    repo_root: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    root = Path(repo_root)
    r_path = root / R22_RELATIVE_PATH
    c_path = root / C22_RELATIVE_PATH
    require(r_path.is_file(), f"R22_MISSING:{r_path}")
    require(c_path.is_file(), f"C22_MISSING:{c_path}")
    require(sha256_file(r_path) == R22_SHA256, "R22_SHA256_MISMATCH")
    require(sha256_file(c_path) == C22_SHA256, "C22_SHA256_MISMATCH")
    r22 = _basis_from_bytes(r_path.read_bytes(), label="R22")
    c22 = _basis_from_bytes(c_path.read_bytes(), label="C22")
    geometry = validate_basis_geometry(r22, c22)
    return r22, c22, geometry


@dataclass(frozen=True)
class CorrectionShape:
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    state_size: int = STATE_SIZE
    rank: int = CORRECTION_RANK

    @property
    def state_width(self) -> int:
        return self.intermediate_size * self.state_size

    @property
    def trainable_numel(self) -> int:
        return self.rank * self.hidden_size + self.state_width * self.rank


FROZEN_SHAPE = CorrectionShape()


def validate_runtime_mixer_dimensions(
    mixer: nn.Module,
    *,
    expected: CorrectionShape = FROZEN_SHAPE,
) -> None:
    observed = CorrectionShape(
        hidden_size=int(getattr(mixer, "hidden_size")),
        intermediate_size=int(getattr(mixer, "intermediate_size")),
        state_size=int(getattr(mixer, "ssm_state_size")),
        rank=expected.rank,
    )
    require(observed.hidden_size == expected.hidden_size, "HIDDEN_SIZE_MISMATCH")
    require(
        observed.intermediate_size == expected.intermediate_size,
        "INTERMEDIATE_SIZE_MISMATCH",
    )
    require(observed.state_size == expected.state_size, "STATE_SIZE_MISMATCH")
    require(observed.state_width == expected.state_width, "STATE_WIDTH_MISMATCH")
    require(
        expected.trainable_numel == EXPECTED_TRAINABLE_NUMEL,
        "FROZEN_TRAINABLE_NUMEL_MISMATCH",
    )


class StateWriteCorrection(nn.Module):
    """Rank-2 bias-free WRITE22 correction with frozen owner/control bases."""

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
            require(shape == FROZEN_SHAPE, "NONFROZEN_CORRECTION_SHAPE")
            require(
                shape.trainable_numel == EXPECTED_TRAINABLE_NUMEL,
                "TRAINABLE_NUMEL_MISMATCH",
            )

        validate_basis_geometry(
            r22,
            c22,
            state_width=shape.state_width,
            rank=shape.rank,
            atol=BASIS_ATOL if strict_frozen_dimensions else 1e-8,
        )

        self.A_theta = nn.Linear(shape.hidden_size, shape.rank, bias=False)
        self.B_theta = nn.Linear(shape.rank, shape.state_width, bias=False)

        # Frozen bases remain buffers, not parameters.
        self.register_buffer("R22", r22.detach().clone(), persistent=True)
        self.register_buffer("C22", c22.detach().clone(), persistent=True)

        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        require(type(seed) is int, "SEED_MUST_BE_INT")
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        # Authority requires CPU deterministic initialization followed by device move.
        a_cpu = torch.empty(
            tuple(self.A_theta.weight.shape),
            dtype=torch.float32,
            device="cpu",
        )
        nn.init.kaiming_uniform_(a_cpu, a=math.sqrt(5), generator=g)
        b_cpu = torch.zeros(
            tuple(self.B_theta.weight.shape),
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
            self.B_theta.weight.copy_(
                b_cpu.to(
                    device=self.B_theta.weight.device,
                    dtype=self.B_theta.weight.dtype,
                )
            )

    def project(self, raw_write: torch.Tensor) -> torch.Tensor:
        require(raw_write.shape[-1] == self.shape.state_width, "WRITE_WIDTH")
        if self.arm == "G5-C0":
            return raw_write

        basis = self.C22 if self.arm == "G5-C1" else self.R22
        basis_live = basis.to(device=raw_write.device, dtype=raw_write.dtype)
        flat = raw_write.reshape(-1, self.shape.state_width)
        coefficients = flat @ basis_live
        removed = coefficients @ basis_live.T
        return (flat - removed).reshape_as(raw_write)

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
        raw = self.B_theta(latent)

        if attention_mask is not None:
            require(
                tuple(attention_mask.shape) == tuple(mixer_input.shape[:2]),
                "ATTENTION_MASK_SHAPE",
            )
            raw = raw * attention_mask.to(dtype=raw.dtype).unsqueeze(-1)

        effective = self.project(raw)
        if return_preproject:
            return raw, effective
        return effective


def reference_correction_recurrence(
    discrete_a: torch.Tensor,
    effective_write: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit autograd-preserving correction recurrence.

    Args:
        discrete_a: [batch, intermediate, seq, state]
        effective_write: [batch, seq, intermediate, state]
        active_mask: optional [batch, seq], masks *new writes* only.

    Returns:
        correction states [batch, seq, intermediate, state]
    """
    require(discrete_a.ndim == 4, "DISCRETE_A_RANK")
    require(effective_write.ndim == 4, "EFFECTIVE_WRITE_RANK")
    b, i, l, n = discrete_a.shape
    require(
        tuple(effective_write.shape) == (b, l, i, n),
        "RECURRENCE_SHAPE_MISMATCH",
    )

    write = effective_write
    if active_mask is not None:
        require(tuple(active_mask.shape) == (b, l), "ACTIVE_MASK_SHAPE")
        write = write * active_mask.to(write.dtype)[:, :, None, None]

    state = torch.zeros(
        (b, i, n),
        dtype=write.dtype,
        device=write.device,
    )
    states: list[torch.Tensor] = []
    for t in range(l):
        state = discrete_a[:, :, t, :].to(write.dtype) * state + write[:, t]
        states.append(state)
    return torch.stack(states, dim=1)


def explicit_total_recurrence(
    discrete_a: torch.Tensor,
    native_write: torch.Tensor,
    correction_write: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verification helper for recurrence superposition."""
    native = reference_correction_recurrence(
        discrete_a,
        native_write,
        active_mask=None,
    )
    correction = reference_correction_recurrence(
        discrete_a,
        correction_write,
        active_mask=active_mask,
    )
    total_direct = reference_correction_recurrence(
        discrete_a,
        native_write
        + (
            correction_write
            if active_mask is None
            else correction_write
            * active_mask.to(correction_write.dtype)[:, :, None, None]
        ),
        active_mask=None,
    )
    return native, correction, total_direct


def reference_correction_mixer_contribution(
    native_mixer: nn.Module,
    mixer_input: torch.Tensor,
    correction: StateWriteCorrection,
    *,
    attention_mask: torch.Tensor | None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute only the additional layer-22 mixer output induced by correction.

    The native mixer forward is *not* replaced here. This branch recomputes the
    frozen input-dependent SSM coefficients and applies the separate correction
    recurrence with no detach on the correction path.
    """
    shape = correction.shape
    require(mixer_input.ndim == 3, "MIXER_INPUT_RANK")
    batch, seq_len, hidden = mixer_input.shape
    require(hidden == shape.hidden_size, "MIXER_INPUT_WIDTH")

    projected = native_mixer.in_proj(mixer_input).transpose(1, 2)
    hidden_states, gate = projected.chunk(2, dim=1)

    if attention_mask is not None:
        require(tuple(attention_mask.shape) == (batch, seq_len), "ATTENTION_MASK_SHAPE")
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(1)
        hidden_states = hidden_states * mask

    # Mirrors Transformers 5.0.0 MambaMixer.slow_forward convolution semantics.
    conv_hidden = native_mixer.act(
        native_mixer.conv1d(hidden_states)[..., :seq_len]
    )

    if attention_mask is not None:
        conv_hidden = conv_hidden * attention_mask.to(conv_hidden.dtype).unsqueeze(1)

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
        native_mixer.dt_proj(time_step)
    ).transpose(1, 2)

    a_continuous = -torch.exp(native_mixer.A_log.float())
    discrete_a = torch.exp(
        a_continuous[None, :, None, :]
        * discrete_time_step[:, :, :, None].float()
    )

    raw, effective = correction(
        mixer_input,
        attention_mask=attention_mask,
        return_preproject=True,
    )
    effective_state_write = effective.reshape(
        batch,
        seq_len,
        shape.intermediate_size,
        shape.state_size,
    )

    corr_states = reference_correction_recurrence(
        discrete_a,
        effective_state_write,
        active_mask=attention_mask,
    )

    # C is [batch, seq, state]. Read out each intermediate channel's state.
    corr_scan = torch.sum(
        corr_states.to(c_readout.dtype) * c_readout[:, :, None, :],
        dim=-1,
    )  # [batch, seq, intermediate]
    corr_scan = corr_scan.transpose(1, 2)
    corr_scan = corr_scan * native_mixer.act(gate)

    # Important: native out_proj bias belongs to the native output and must not
    # be added a second time to the correction contribution.
    contribution = F.linear(
        corr_scan.transpose(1, 2),
        native_mixer.out_proj.weight,
        bias=None,
    )

    if not return_diagnostics:
        return contribution

    diagnostics = {
        "raw_write": raw,
        "effective_write": effective,
        "correction_states": corr_states,
        "correction_scan": corr_scan.transpose(1, 2),
    }
    return contribution, diagnostics


class Phase2Layer22MixerWrapper(nn.Module):
    """Owns the exact frozen native mixer plus a separate correction branch."""

    def __init__(
        self,
        *,
        native_mixer: nn.Module,
        correction: StateWriteCorrection,
        strict_frozen_dimensions: bool = True,
    ) -> None:
        super().__init__()
        if strict_frozen_dimensions:
            validate_runtime_mixer_dimensions(native_mixer)
        self.native_mixer = native_mixer
        self.correction = correction
        self.layer_idx = int(getattr(native_mixer, "layer_idx", TARGET_LAYER))
        self._phase2_active_mask: torch.Tensor | None = None
        require(
            self.layer_idx == TARGET_LAYER or not strict_frozen_dimensions,
            f"TARGET_LAYER_MISMATCH:{self.layer_idx}",
        )

    @contextmanager
    def active_mask(self, attention_mask: torch.Tensor) -> Iterator[None]:
        """Supply the original controlled-data attention mask to WRITE22.

        ContraMambaV6BMinimal historically calls ``self.mamba(input_ids=...)``
        without forwarding its attention mask. Phase 2 therefore supplies the
        mask out-of-band at the wrapper boundary rather than editing the
        historical model file.
        """
        require(self._phase2_active_mask is None, "PHASE2_ACTIVE_MASK_REENTRANT")
        require(attention_mask.ndim == 2, "PHASE2_ACTIVE_MASK_RANK")
        self._phase2_active_mask = attention_mask
        try:
            yield
        finally:
            self._phase2_active_mask = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Scientific training path is full-sequence, no-cache only.
        require(cache_params is None, "PHASE2_REFERENCE_BACKEND_CACHE_UNSUPPORTED")
        require(cache_position is None, "PHASE2_REFERENCE_BACKEND_CACHE_POSITION_UNSUPPORTED")

        effective_mask = (
            attention_mask
            if attention_mask is not None
            else self._phase2_active_mask
        )
        require(effective_mask is not None, "PHASE2_ACTIVE_MASK_REQUIRED")
        require(
            tuple(effective_mask.shape) == tuple(hidden_states.shape[:2]),
            "PHASE2_ACTIVE_MASK_SHAPE",
        )

        # Preserve the exact historical native call semantics. In the frozen
        # ContraMambaV6BMinimal path the Mamba backbone receives no attention
        # mask, so the wrapper must not silently change the native mixer call.
        native_output = self.native_mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        correction_output = reference_correction_mixer_contribution(
            self.native_mixer,
            hidden_states,
            self.correction,
            attention_mask=effective_mask,
        )
        return native_output + correction_output


def _logical_parent_name(name: str) -> str:
    return name.replace(".mixer.native_mixer.", ".mixer.")


def parent_parameter_fingerprint(model: nn.Module) -> str:
    """Fingerprint historical parent parameter values with wrapper-name normalization."""
    h = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if ".correction." in name:
            continue
        logical = _logical_parent_name(name)
        rows.append((logical, parameter))
    rows.sort(key=lambda item: item[0])

    for logical, parameter in rows:
        value = parameter.detach().cpu().contiguous()
        h.update(logical.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(value.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(value.numpy().tobytes())
        h.update(b"\n")
    return h.hexdigest()


def correction_parameter_audit(model: nn.Module) -> dict[str, Any]:
    trainable = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    names = [name for name, _ in trainable]
    numel = sum(p.numel() for _, p in trainable)
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_numel": int(numel),
        "trainable_names": names,
        "all_trainable_are_correction": all(".correction." in name for name in names),
    }


def freeze_parent_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def install_phase2_layer22_wrapper(
    model: nn.Module,
    *,
    arm: str,
    r22: torch.Tensor,
    c22: torch.Tensor,
    seed: int,
) -> Phase2Layer22MixerWrapper:
    """Freeze parent, attach one layer-22 wrapper, and expose only A/B as trainable."""
    validate_arm(arm)
    require(hasattr(model, "mamba"), "MODEL_MAMBA_MISSING")
    require(hasattr(model.mamba, "layers"), "MAMBA_LAYERS_MISSING")
    require(len(model.mamba.layers) > TARGET_LAYER, "LAYER22_MISSING")

    target_block = model.mamba.layers[TARGET_LAYER]
    native_mixer = target_block.mixer
    require(
        not isinstance(native_mixer, Phase2Layer22MixerWrapper),
        "LAYER22_ALREADY_WRAPPED",
    )
    validate_runtime_mixer_dimensions(native_mixer)

    before = parent_parameter_fingerprint(model)
    freeze_parent_parameters(model)

    native_weight = native_mixer.in_proj.weight
    correction = StateWriteCorrection(
        arm=arm,
        r22=r22,
        c22=c22,
        seed=seed,
    )
    correction.to(
        device=native_weight.device,
        dtype=native_weight.dtype,
    )
    # Restore basis precision after module dtype conversion.
    correction.R22 = r22.detach().clone().to(
        device=native_weight.device,
        dtype=torch.float64,
    )
    correction.C22 = c22.detach().clone().to(
        device=native_weight.device,
        dtype=torch.float64,
    )

    wrapper = Phase2Layer22MixerWrapper(
        native_mixer=native_mixer,
        correction=correction,
    )
    target_block.mixer = wrapper

    after = parent_parameter_fingerprint(model)
    require(before == after, "PARENT_PARAMETER_FINGERPRINT_CHANGED_ON_INSTALL")

    audit = correction_parameter_audit(model)
    require(audit["trainable_tensor_count"] == 2, "TRAINABLE_TENSOR_COUNT")
    require(audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL, "TRAINABLE_NUMEL")
    require(audit["all_trainable_are_correction"], "NONCORRECTION_TRAINABLE_PARAMETER")
    return wrapper


def remove_phase2_layer22_wrapper(model: nn.Module) -> nn.Module:
    require(hasattr(model, "mamba"), "MODEL_MAMBA_MISSING")
    block = model.mamba.layers[TARGET_LAYER]
    wrapper = block.mixer
    require(
        isinstance(wrapper, Phase2Layer22MixerWrapper),
        "LAYER22_NOT_WRAPPED",
    )
    native = wrapper.native_mixer
    block.mixer = native
    return native


def get_phase2_layer22_wrapper(model: nn.Module) -> Phase2Layer22MixerWrapper:
    require(hasattr(model, "mamba"), "MODEL_MAMBA_MISSING")
    wrapper = model.mamba.layers[TARGET_LAYER].mixer
    require(
        isinstance(wrapper, Phase2Layer22MixerWrapper),
        "LAYER22_NOT_WRAPPED",
    )
    return wrapper


@contextmanager
def phase2_active_mask(
    model: nn.Module,
    attention_mask: torch.Tensor,
) -> Iterator[None]:
    """Out-of-band active-mask bridge for historical V6B forward semantics."""
    wrapper = get_phase2_layer22_wrapper(model)
    with wrapper.active_mask(attention_mask):
        yield


def correction_optimizer_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    audit = correction_parameter_audit(model)
    require(audit["trainable_tensor_count"] == 2, "OPTIMIZER_TENSOR_COUNT")
    require(audit["all_trainable_are_correction"], "OPTIMIZER_OWNERSHIP")
    parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and ".correction." in name
    ]
    require(len(parameters) == 2, "OPTIMIZER_PARAMETER_LIST")
    return parameters


def phase2_final_three_way_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    require(logits.ndim == 2 and logits.shape[-1] == 3, "FINAL_LOGITS_SHAPE")
    require(labels.ndim == 1 and labels.shape[0] == logits.shape[0], "FINAL_LABEL_SHAPE")
    return F.cross_entropy(logits, labels)
