"""Gen5 Stage E learned-B-plane A-fixed M-only diagnostic primitives.

Design + implementation authority:
    9fa2af57414519ccc8fefe44ed2fc89d98a25626

Importing this module does not authorize CUDA, training, evaluation, backward,
optimizer construction, or optimizer steps.
"""

from __future__ import annotations

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
    StageELayer22MixerWrapper,
)
from contramamba import gen5_stage_e_learned_b_plane_ainit_control as ainit


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "9fa2af57414519ccc8fefe44ed2fc89d98a25626"
)
ARM = "E-BFREE-AFIX-MONLY"
TRAINING_SEEDS = ainit.TRAINING_SEEDS
SOURCE_CORRECTIONS = ainit.SOURCE_CORRECTIONS
EXPECTED_TRAINABLE_NUMEL = 4


class LearnedBPlaneAFixMOnlyError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LearnedBPlaneAFixMOnlyError(message)


def tensor_sha256(value: torch.Tensor) -> str:
    return ainit.tensor_sha256(value)


def load_seed_matched_afix_monly_source(
    repo_root: str | Path,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    return ainit.load_seed_matched_ainit_source(repo_root, seed)


class LearnedBPlaneAFixMOnlyCorrection(ainit.LearnedBPlaneAInitCorrection):
    """AINIT QMA correction with exact A_free frozen and only M trainable."""

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
            a_free=a_free,
            seed=seed,
            shape=shape,
            strict_frozen_dimensions=strict_frozen_dimensions,
        )
        self.arm = ARM
        self.A_theta.weight.requires_grad_(False)
        self.M_theta.weight.requires_grad_(True)
        require(
            not self.A_theta.weight.requires_grad,
            "AFIX_A_REQUIRES_GRAD",
        )
        require(
            self.M_theta.weight.requires_grad,
            "AFIX_M_NOT_TRAINABLE",
        )
        require(
            int(torch.count_nonzero(self.M_theta.weight).item()) == 0,
            "AFIX_M_NONZERO",
        )


def install_learned_b_plane_afix_monly_layer22_wrapper(
    model: nn.Module,
    *,
    q_bfree: torch.Tensor,
    a_free: torch.Tensor,
    seed: int,
    shape: CorrectionShape = FROZEN_SHAPE,
    strict_frozen_dimensions: bool = True,
) -> StageELayer22MixerWrapper:
    before = parent_parameter_fingerprint(model)
    wrapper = ainit.install_learned_b_plane_ainit_layer22_wrapper(
        model,
        q_bfree=q_bfree,
        a_free=a_free,
        seed=seed,
        shape=shape,
        strict_frozen_dimensions=strict_frozen_dimensions,
    )
    correction = wrapper.correction
    correction.arm = ARM
    correction.A_theta.weight.requires_grad_(False)
    correction.M_theta.weight.requires_grad_(True)

    source_a = a_free.detach().cpu().contiguous()
    require(
        tensor_sha256(correction.A_theta.weight)
        == tensor_sha256(
            source_a.to(dtype=correction.A_theta.weight.dtype)
        ),
        "AFIX_INSTALL_A_IDENTITY",
    )
    require(
        not correction.A_theta.weight.requires_grad,
        "AFIX_INSTALL_A_REQUIRES_GRAD",
    )
    require(
        correction.M_theta.weight.requires_grad,
        "AFIX_INSTALL_M_NOT_TRAINABLE",
    )
    require(
        int(torch.count_nonzero(correction.M_theta.weight).item()) == 0,
        "AFIX_INSTALL_M_NONZERO",
    )
    after = parent_parameter_fingerprint(model)
    require(before == after, "AFIX_PARENT_FINGERPRINT_CHANGED")
    return wrapper


def zero_output_firewall(correction: nn.Module) -> dict[str, Any]:
    result = ainit.zero_output_firewall(correction)
    require(
        not correction.A_theta.weight.requires_grad,
        "AFIX_FIREWALL_A_REQUIRES_GRAD",
    )
    require(
        correction.M_theta.weight.requires_grad,
        "AFIX_FIREWALL_M_NOT_TRAINABLE",
    )
    return {
        **result,
        "A_requires_grad": False,
        "M_requires_grad": True,
    }


def trainable_parameter_audit(correction: nn.Module) -> dict[str, Any]:
    rows = [
        (name, parameter)
        for name, parameter in correction.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in rows]
    require(
        names == ["M_theta.weight"],
        f"AFIX_TRAINABLE_NAMES:{names}",
    )
    numel = sum(parameter.numel() for _, parameter in rows)
    require(
        numel == EXPECTED_TRAINABLE_NUMEL,
        f"AFIX_TRAINABLE_NUMEL:{numel}",
    )
    require(
        not correction.A_theta.weight.requires_grad,
        "AFIX_A_REQUIRES_GRAD_AUDIT",
    )
    return {
        "trainable_tensor_names": names,
        "trainable_tensor_count": len(rows),
        "trainable_numel": int(numel),
        "A_requires_grad": False,
        "M_requires_grad": True,
    }
