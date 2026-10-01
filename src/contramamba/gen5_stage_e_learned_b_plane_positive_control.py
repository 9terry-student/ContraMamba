"""Gen5 Stage E learned-B-plane positive-control primitives.

Design + implementation authority:
    c9ef6c55448a3458b46f5b76b5c88244cc7b726e

This module implements only the seed-matched learned-output-plane positive
control.  Importing it does not authorize CUDA, training, evaluation, backward,
or optimizer steps.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from contramamba.gen5_phase2_state_update_ownership import (
    FROZEN_SHAPE,
    CorrectionShape,
    freeze_parent_parameters,
    parent_parameter_fingerprint,
    validate_runtime_mixer_dimensions,
)
from contramamba.gen5_stage_e_causal_plane_bottleneck import (
    EXPECTED_TRAINABLE_NUMEL,
    StageELayer22MixerWrapper,
)


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "c9ef6c55448a3458b46f5b76b5c88244cc7b726e"
)
ARM = "E-BFREE"
TRAINING_SEEDS = (6201, 6202, 6203)
QR_ORTH_ATOL = 1e-12
QR_RECON_REL_ATOL = 1e-12
FIXED_PLANE_RESIDUAL_ATOL = 5e-6


class LearnedBPlaneError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LearnedBPlaneError(message)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class SourceCorrectionSpec:
    seed: int
    relative_path: str
    file_sha256: str


SOURCE_CORRECTIONS = {
    6201: SourceCorrectionSpec(
        seed=6201,
        relative_path=(
            "reports/reason_router_gen5_phase3a_training_runs/"
            "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
            "cells/seed6201/P0/final_correction.pt"
        ),
        file_sha256=(
            "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf"
        ),
    ),
    6202: SourceCorrectionSpec(
        seed=6202,
        relative_path=(
            "reports/reason_router_gen5_phase3a_training_runs/"
            "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
            "cells/seed6202/P0/final_correction.pt"
        ),
        file_sha256=(
            "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214"
        ),
    ),
    6203: SourceCorrectionSpec(
        seed=6203,
        relative_path=(
            "reports/reason_router_gen5_phase3a_training_runs/"
            "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
            "cells/seed6203/P0/final_correction.pt"
        ),
        file_sha256=(
            "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770"
        ),
    ),
}

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)


def _canonical_qr(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Thin float64 QR with deterministic positive-diagonal sign convention."""
    value = matrix.detach().cpu().to(torch.float64).contiguous()
    require(value.ndim == 2, "BFREE_MATRIX_RANK")
    require(tuple(value.shape) == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank), "BFREE_MATRIX_SHAPE")
    require(bool(torch.isfinite(value).all().item()), "BFREE_MATRIX_NONFINITE")
    require(int(torch.linalg.matrix_rank(value).item()) == FROZEN_SHAPE.rank, "BFREE_MATRIX_RANK_NOT_TWO")

    q, r = torch.linalg.qr(value, mode="reduced")
    diagonal = torch.diagonal(r).clone()
    require(bool(torch.all(diagonal != 0).item()), "QR_ZERO_DIAGONAL")

    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    r = signs.unsqueeze(1) * r

    require(bool(torch.all(torch.diagonal(r) >= 0).item()), "QR_SIGN_CANONICALIZATION")
    identity = torch.eye(FROZEN_SHAPE.rank, dtype=torch.float64)
    orth_max = float(torch.max(torch.abs(q.T @ q - identity)).item())
    require(orth_max <= QR_ORTH_ATOL, f"QR_ORTHOGONALITY:{orth_max}")

    reconstruction = q @ r
    recon_fro = float(torch.linalg.matrix_norm(reconstruction - value).item())
    source_fro = float(torch.linalg.matrix_norm(value).item())
    recon_rel = recon_fro / source_fro if source_fro > 0.0 else float("inf")
    require(recon_rel <= QR_RECON_REL_ATOL, f"QR_RECONSTRUCTION_REL:{recon_rel}")

    projector_residual = value - q @ (q.T @ value)
    projector_rel = float(torch.linalg.matrix_norm(projector_residual).item()) / source_fro
    require(projector_rel <= QR_RECON_REL_ATOL, f"QR_PROJECTOR_RESIDUAL_REL:{projector_rel}")
    return q.contiguous(), r.contiguous()


def load_seed_matched_learned_plane(
    repo_root: str | Path,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    require(seed in SOURCE_CORRECTIONS, f"SEED:{seed}")
    spec = SOURCE_CORRECTIONS[seed]
    path = Path(repo_root) / spec.relative_path
    require(path.is_file(), f"BFREE_SOURCE_MISSING:{spec.relative_path}")

    observed_file_sha = sha256_file(path)
    require(
        observed_file_sha == spec.file_sha256,
        f"BFREE_SOURCE_SHA256:{seed}:{observed_file_sha}",
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "BFREE_PAYLOAD_TYPE")
    require(payload.get("schema_version") == "GEN5_PHASE3A_FINAL_CORRECTION_V1", "BFREE_SCHEMA")
    require(int(payload.get("seed", -1)) == seed, "BFREE_SEED")
    require(payload.get("arm") == "G5-C0", "BFREE_ARM")
    require(payload.get("pressure") == "P0", "BFREE_PRESSURE")
    require(
        payload.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256,
        "BFREE_PARENT_CHECKPOINT",
    )

    state_dict = payload.get("state_dict")
    require(isinstance(state_dict, dict), "BFREE_STATE_DICT")
    b = state_dict.get("B_theta.weight")
    require(torch.is_tensor(b), "BFREE_B_TENSOR_MISSING")
    b = b.detach().cpu().contiguous()
    require(tuple(b.shape) == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank), "BFREE_B_SHAPE")
    require(bool(torch.isfinite(b).all().item()), "BFREE_B_NONFINITE")
    require(int(torch.linalg.matrix_rank(b.to(torch.float64)).item()) == FROZEN_SHAPE.rank, "BFREE_B_RANK")

    tensor_hashes = payload.get("tensor_sha256")
    if isinstance(tensor_hashes, dict) and "B_theta.weight" in tensor_hashes:
        require(
            tensor_sha256(b) == str(tensor_hashes["B_theta.weight"]),
            "BFREE_B_TENSOR_SHA256",
        )

    q, r = _canonical_qr(b)
    b64 = b.to(torch.float64)
    reconstruction = q @ r
    source_fro = float(torch.linalg.matrix_norm(b64).item())
    reconstruction_rel = (
        float(torch.linalg.matrix_norm(reconstruction - b64).item()) / source_fro
    )
    projection_rel = (
        float(torch.linalg.matrix_norm(b64 - q @ (q.T @ b64)).item()) / source_fro
    )

    metadata = {
        "seed": seed,
        "source_relative_path": spec.relative_path,
        "source_file_sha256": observed_file_sha,
        "source_B_tensor_sha256": tensor_sha256(b),
        "source_B_rank": int(torch.linalg.matrix_rank(b64).item()),
        "Q_tensor_sha256": tensor_sha256(q),
        "R_tensor_sha256": tensor_sha256(r),
        "qr_reconstruction_relative": reconstruction_rel,
        "projector_residual_relative": projection_rel,
        "q_orthogonality_max_abs": float(
            torch.max(
                torch.abs(
                    q.T @ q
                    - torch.eye(FROZEN_SHAPE.rank, dtype=torch.float64)
                )
            ).item()
        ),
    }
    return q, metadata


def load_all_seed_matched_learned_planes(
    repo_root: str | Path,
) -> dict[int, tuple[torch.Tensor, dict[str, Any]]]:
    return {
        seed: load_seed_matched_learned_plane(repo_root, seed)
        for seed in TRAINING_SEEDS
    }


class LearnedBPlaneCorrection(nn.Module):
    """Stage-E QMA correction with seed-matched frozen learned output plane."""

    def __init__(
        self,
        *,
        q_bfree: torch.Tensor,
        seed: int,
        shape: CorrectionShape = FROZEN_SHAPE,
        strict_frozen_dimensions: bool = True,
    ) -> None:
        super().__init__()
        require(seed in TRAINING_SEEDS, f"SEED:{seed}")
        if strict_frozen_dimensions:
            require(shape == FROZEN_SHAPE, "NONFROZEN_BFREE_SHAPE")
            require(
                shape.rank * shape.hidden_size + shape.rank * shape.rank
                == EXPECTED_TRAINABLE_NUMEL,
                "BFREE_TRAINABLE_NUMEL_CONTRACT",
            )

        q = q_bfree.detach().cpu().to(torch.float64).contiguous()
        require(tuple(q.shape) == (shape.state_width, shape.rank), "Q_BFREE_SHAPE")
        require(bool(torch.isfinite(q).all().item()), "Q_BFREE_NONFINITE")
        identity = torch.eye(shape.rank, dtype=torch.float64)
        orth_max = float(torch.max(torch.abs(q.T @ q - identity)).item())
        require(orth_max <= QR_ORTH_ATOL, f"Q_BFREE_ORTHOGONALITY:{orth_max}")

        self.shape = shape
        self.seed = seed
        self.arm = ARM
        self.A_theta = nn.Linear(shape.hidden_size, shape.rank, bias=False)
        self.M_theta = nn.Linear(shape.rank, shape.rank, bias=False)
        self.register_buffer("Q_BFREE", q.clone(), persistent=True)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        # Exact Stage E initialization convention.
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
        return self.Q_BFREE

    def effective_output_matrix(
        self,
        m_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = self.M_theta.weight if m_weight is None else m_weight
        require(tuple(m.shape) == (self.shape.rank, self.shape.rank), "M_WEIGHT_SHAPE")
        q = self.Q_BFREE.to(device=m.device, dtype=m.dtype)
        return q @ m

    def forward(
        self,
        mixer_input: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        require(mixer_input.ndim == 3, "MIXER_INPUT_RANK")
        require(mixer_input.shape[-1] == self.shape.hidden_size, "MIXER_INPUT_WIDTH")
        latent = self.A_theta(mixer_input)
        mixed = self.M_theta(latent)
        q = self.Q_BFREE.to(device=mixed.device, dtype=mixed.dtype)
        effective = torch.nn.functional.linear(mixed, q, bias=None)
        if attention_mask is not None:
            require(
                tuple(attention_mask.shape) == tuple(mixer_input.shape[:2]),
                "ATTENTION_MASK_SHAPE",
            )
            effective = effective * attention_mask.to(effective.dtype).unsqueeze(-1)
        return effective


def learned_b_fixed_plane_geometry(
    correction: LearnedBPlaneCorrection,
) -> dict[str, float | int]:
    q = correction.Q_BFREE.detach().cpu().to(torch.float64)
    m = correction.M_theta.weight.detach().cpu().to(torch.float64)
    a = correction.A_theta.weight.detach().cpu().to(torch.float64)
    b_eff = q @ m
    projected = q @ (q.T @ b_eff)
    residual = b_eff - projected

    b_norm = float(torch.linalg.matrix_norm(b_eff).item())
    residual_fro = float(torch.linalg.matrix_norm(residual).item())
    return {
        "effective_output_rank": int(torch.linalg.matrix_rank(b_eff).item()),
        "effective_operator_rank": int(torch.linalg.matrix_rank(m @ a).item()),
        "effective_output_frobenius_norm": b_norm,
        "fixed_plane_residual_frobenius": residual_fro,
        "fixed_plane_residual_max_abs": float(torch.max(torch.abs(residual)).item()),
        "fixed_plane_residual_relative": residual_fro / b_norm if b_norm > 0 else 0.0,
        "q_orthogonality_max_abs": float(
            torch.max(
                torch.abs(
                    q.T @ q
                    - torch.eye(q.shape[1], dtype=torch.float64)
                )
            ).item()
        ),
    }


def install_learned_b_plane_layer22_wrapper(
    model: nn.Module,
    *,
    q_bfree: torch.Tensor,
    seed: int,
    shape: CorrectionShape = FROZEN_SHAPE,
    strict_frozen_dimensions: bool = True,
) -> StageELayer22MixerWrapper:
    require(hasattr(model, "mamba"), "MODEL_MAMBA_MISSING")
    require(hasattr(model.mamba, "layers"), "MAMBA_LAYERS_MISSING")
    require(len(model.mamba.layers) > 22, "LAYER22_MISSING")

    target_block = model.mamba.layers[22]
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
    correction = LearnedBPlaneCorrection(
        q_bfree=q_bfree,
        seed=seed,
        shape=shape,
        strict_frozen_dimensions=strict_frozen_dimensions,
    )
    correction.to(
        device=native_weight.device,
        dtype=native_weight.dtype,
    )
    # Preserve exact authenticated Q in float64, matching Stage E basis handling.
    correction.Q_BFREE = q_bfree.detach().clone().to(
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
    require(before == after, "PARENT_PARAMETER_FINGERPRINT_CHANGED_ON_INSTALL")
    return wrapper
