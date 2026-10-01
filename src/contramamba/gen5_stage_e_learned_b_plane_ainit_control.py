"""Gen5 Stage E learned-B-plane A-initialization diagnostic primitives.

Design + implementation authority:
    c24bc199b6b8fb94002c0563535ff7b5284794b2

Importing this module does not authorize CUDA, training, evaluation, backward,
or optimizer steps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from contramamba.gen5_phase2_state_update_ownership import (
    FROZEN_SHAPE,
    CorrectionShape,
    parent_parameter_fingerprint,
)
from contramamba.gen5_stage_e_causal_plane_bottleneck import (
    EXPECTED_TRAINABLE_NUMEL,
    StageELayer22MixerWrapper,
)
from contramamba import gen5_stage_e_learned_b_plane_positive_control as bfree


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "c24bc199b6b8fb94002c0563535ff7b5284794b2"
)
ARM = "E-BFREE-AINIT"
TRAINING_SEEDS = bfree.TRAINING_SEEDS
SOURCE_CORRECTIONS = bfree.SOURCE_CORRECTIONS
REPRESENTABILITY_REL_ATOL = 1e-12


class LearnedBPlaneAInitError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LearnedBPlaneAInitError(message)


def tensor_sha256(value: torch.Tensor) -> str:
    return bfree.tensor_sha256(value)


def _load_authenticated_source_payload(
    repo_root: str | Path,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    require(seed in SOURCE_CORRECTIONS, f"SEED:{seed}")
    spec = SOURCE_CORRECTIONS[seed]
    path = Path(repo_root) / spec.relative_path
    require(path.is_file(), f"AINIT_SOURCE_MISSING:{spec.relative_path}")
    observed_file_sha = bfree.sha256_file(path)
    require(
        observed_file_sha == spec.file_sha256,
        f"AINIT_SOURCE_SHA256:{seed}:{observed_file_sha}",
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "AINIT_PAYLOAD_TYPE")
    require(
        payload.get("schema_version") == "GEN5_PHASE3A_FINAL_CORRECTION_V1",
        "AINIT_SCHEMA",
    )
    require(int(payload.get("seed", -1)) == seed, "AINIT_SEED")
    require(payload.get("arm") == "G5-C0", "AINIT_ARM")
    require(payload.get("pressure") == "P0", "AINIT_PRESSURE")
    require(
        payload.get("parent_checkpoint_sha256")
        == bfree.PARENT_CHECKPOINT_SHA256,
        "AINIT_PARENT_CHECKPOINT",
    )

    state_dict = payload.get("state_dict")
    require(isinstance(state_dict, dict), "AINIT_STATE_DICT")
    require(
        "A_theta.weight" in state_dict and "B_theta.weight" in state_dict,
        "AINIT_REQUIRED_TENSORS",
    )
    a = state_dict["A_theta.weight"]
    b = state_dict["B_theta.weight"]
    require(torch.is_tensor(a), "AINIT_A_TENSOR_MISSING")
    require(torch.is_tensor(b), "AINIT_B_TENSOR_MISSING")
    a = a.detach().cpu().contiguous()
    b = b.detach().cpu().contiguous()

    require(
        tuple(a.shape) == (FROZEN_SHAPE.rank, FROZEN_SHAPE.hidden_size),
        f"AINIT_A_SHAPE:{tuple(a.shape)}",
    )
    require(
        tuple(b.shape) == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank),
        f"AINIT_B_SHAPE:{tuple(b.shape)}",
    )
    require(bool(torch.isfinite(a).all().item()), "AINIT_A_NONFINITE")
    require(bool(torch.isfinite(b).all().item()), "AINIT_B_NONFINITE")
    a_rank = int(torch.linalg.matrix_rank(a.to(torch.float64)).item())
    b_rank = int(torch.linalg.matrix_rank(b.to(torch.float64)).item())
    require(b_rank == FROZEN_SHAPE.rank, f"AINIT_B_RANK:{b_rank}")

    tensor_hashes = payload.get("tensor_sha256")
    if isinstance(tensor_hashes, dict):
        if "A_theta.weight" in tensor_hashes:
            require(
                tensor_sha256(a) == str(tensor_hashes["A_theta.weight"]),
                "AINIT_A_TENSOR_SHA256",
            )
        if "B_theta.weight" in tensor_hashes:
            require(
                tensor_sha256(b) == str(tensor_hashes["B_theta.weight"]),
                "AINIT_B_TENSOR_SHA256",
            )

    return a, b, {
        "seed": seed,
        "source_relative_path": spec.relative_path,
        "source_file_sha256": observed_file_sha,
        "source_A_tensor_sha256": tensor_sha256(a),
        "source_B_tensor_sha256": tensor_sha256(b),
        "source_A_row_rank": a_rank,
        "source_B_rank": b_rank,
    }


def factorized_operator_representability(
    *,
    a_free: torch.Tensor,
    b_free: torch.Tensor,
    q_bfree: torch.Tensor,
    r_bfree: torch.Tensor,
) -> dict[str, float]:
    a = a_free.detach().cpu().to(torch.float64).contiguous()
    b = b_free.detach().cpu().to(torch.float64).contiguous()
    q = q_bfree.detach().cpu().to(torch.float64).contiguous()
    r = r_bfree.detach().cpu().to(torch.float64).contiguous()

    require(
        tuple(a.shape) == (FROZEN_SHAPE.rank, FROZEN_SHAPE.hidden_size),
        "REP_A_SHAPE",
    )
    require(
        tuple(b.shape) == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank),
        "REP_B_SHAPE",
    )
    require(
        tuple(q.shape) == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank),
        "REP_Q_SHAPE",
    )
    require(
        tuple(r.shape) == (FROZEN_SHAPE.rank, FROZEN_SHAPE.rank),
        "REP_R_SHAPE",
    )

    d = b - q @ r
    g_a = a @ a.T
    residual_sq_tensor = torch.trace((d.T @ d) @ g_a)
    denominator_sq_tensor = torch.trace((b.T @ b) @ g_a)

    residual_sq = float(residual_sq_tensor.item())
    denominator_sq = float(denominator_sq_tensor.item())
    require(math.isfinite(residual_sq), "REP_RESIDUAL_NONFINITE")
    require(math.isfinite(denominator_sq), "REP_DENOMINATOR_NONFINITE")
    require(denominator_sq > 0.0, f"REP_DENOMINATOR:{denominator_sq}")

    # A tiny negative value could only be numerical roundoff in the trace.
    require(
        residual_sq >= -1e-24 * denominator_sq,
        f"REP_RESIDUAL_NEGATIVE:{residual_sq}",
    )
    residual_sq = max(0.0, residual_sq)
    relative = math.sqrt(residual_sq / denominator_sq)
    require(
        relative <= REPRESENTABILITY_REL_ATOL,
        f"REPRESENTABILITY_RELATIVE:{relative}",
    )
    return {
        "factorized_operator_residual_sq": residual_sq,
        "factorized_operator_denominator_sq": denominator_sq,
        "factorized_operator_representability_relative": relative,
    }


def load_seed_matched_ainit_source(
    repo_root: str | Path,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    a_free, b_free, auth = _load_authenticated_source_payload(repo_root, seed)
    q, r = bfree._canonical_qr(b_free)
    b64 = b_free.to(torch.float64)
    source_fro = float(torch.linalg.matrix_norm(b64).item())
    qr_recon_rel = (
        float(torch.linalg.matrix_norm(q @ r - b64).item()) / source_fro
    )
    rep = factorized_operator_representability(
        a_free=a_free,
        b_free=b_free,
        q_bfree=q,
        r_bfree=r,
    )
    metadata = {
        **auth,
        "Q_tensor_sha256": tensor_sha256(q),
        "R_tensor_sha256": tensor_sha256(r),
        "qr_reconstruction_relative": qr_recon_rel,
        **rep,
    }
    return q, a_free, metadata


class LearnedBPlaneAInitCorrection(bfree.LearnedBPlaneCorrection):
    """BFREE QMA correction initialized with seed-matched unrestricted A."""

    def __init__(
        self,
        *,
        q_bfree: torch.Tensor,
        a_free: torch.Tensor,
        seed: int,
        shape: CorrectionShape = FROZEN_SHAPE,
        strict_frozen_dimensions: bool = True,
    ) -> None:
        super().__init__(
            q_bfree=q_bfree,
            seed=seed,
            shape=shape,
            strict_frozen_dimensions=strict_frozen_dimensions,
        )
        require(seed in TRAINING_SEEDS, f"SEED:{seed}")
        a = a_free.detach().cpu().contiguous()
        require(
            tuple(a.shape) == tuple(self.A_theta.weight.shape),
            f"AINIT_COPY_SHAPE:{tuple(a.shape)}",
        )
        require(bool(torch.isfinite(a).all().item()), "AINIT_COPY_NONFINITE")
        with torch.no_grad():
            self.A_theta.weight.copy_(
                a.to(
                    device=self.A_theta.weight.device,
                    dtype=self.A_theta.weight.dtype,
                )
            )
            self.M_theta.weight.zero_()
        self.arm = ARM
        require(
            torch.equal(
                self.A_theta.weight.detach().cpu(),
                a.to(dtype=self.A_theta.weight.dtype),
            ),
            "AINIT_COPY_IDENTITY",
        )
        require(
            int(torch.count_nonzero(self.M_theta.weight).item()) == 0,
            "AINIT_M_NONZERO",
        )


def install_learned_b_plane_ainit_layer22_wrapper(
    model: nn.Module,
    *,
    q_bfree: torch.Tensor,
    a_free: torch.Tensor,
    seed: int,
    shape: CorrectionShape = FROZEN_SHAPE,
    strict_frozen_dimensions: bool = True,
) -> StageELayer22MixerWrapper:
    wrapper = bfree.install_learned_b_plane_layer22_wrapper(
        model,
        q_bfree=q_bfree,
        seed=seed,
        shape=shape,
        strict_frozen_dimensions=strict_frozen_dimensions,
    )
    before = parent_parameter_fingerprint(model)
    correction = wrapper.correction
    a = a_free.detach().cpu().contiguous()
    require(
        tuple(a.shape) == tuple(correction.A_theta.weight.shape),
        "AINIT_INSTALL_A_SHAPE",
    )
    with torch.no_grad():
        correction.A_theta.weight.copy_(
            a.to(
                device=correction.A_theta.weight.device,
                dtype=correction.A_theta.weight.dtype,
            )
        )
        correction.M_theta.weight.zero_()
    correction.arm = ARM
    require(
        int(torch.count_nonzero(correction.M_theta.weight).item()) == 0,
        "AINIT_INSTALL_M_NONZERO",
    )
    require(
        tensor_sha256(correction.A_theta.weight)
        == tensor_sha256(a.to(dtype=correction.A_theta.weight.dtype)),
        "AINIT_INSTALL_A_IDENTITY",
    )
    after = parent_parameter_fingerprint(model)
    require(before == after, "AINIT_PARENT_FINGERPRINT_CHANGED")
    return wrapper


def zero_output_firewall(
    correction: nn.Module,
) -> dict[str, Any]:
    require(
        int(torch.count_nonzero(correction.M_theta.weight).item()) == 0,
        "AINIT_FIREWALL_M_NONZERO",
    )
    x = torch.linspace(
        -1.0,
        1.0,
        steps=2 * FROZEN_SHAPE.hidden_size,
        dtype=correction.A_theta.weight.dtype,
        device=correction.A_theta.weight.device,
    ).reshape(1, 2, FROZEN_SHAPE.hidden_size)
    with torch.no_grad():
        y = correction(x)
    nonzero = int(torch.count_nonzero(y).item())
    require(nonzero == 0, f"AINIT_FIREWALL_OUTPUT_NONZERO:{nonzero}")
    require(bool(torch.isfinite(y).all().item()), "AINIT_FIREWALL_OUTPUT_NONFINITE")
    return {
        "M_nonzero_count": 0,
        "correction_output_nonzero_count": 0,
        "correction_output_exact_zero": True,
    }


def trainable_parameter_audit(
    correction: nn.Module,
) -> dict[str, Any]:
    rows = [
        (name, parameter)
        for name, parameter in correction.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in rows]
    require(
        names == ["A_theta.weight", "M_theta.weight"],
        f"AINIT_TRAINABLE_NAMES:{names}",
    )
    numel = sum(parameter.numel() for _, parameter in rows)
    require(
        numel == EXPECTED_TRAINABLE_NUMEL,
        f"AINIT_TRAINABLE_NUMEL:{numel}",
    )
    return {
        "trainable_tensor_names": names,
        "trainable_tensor_count": len(rows),
        "trainable_numel": int(numel),
    }
