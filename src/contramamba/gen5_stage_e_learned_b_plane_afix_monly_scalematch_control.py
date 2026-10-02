"""Gen5 Stage E learned-B-plane A-fixed M-only scale-matched primitives.

Design + implementation authority:
    1ea8d156c86a7f8ec6b9ad638632e46b54b0de38

This module changes no scientific quantity except the arm identity used by the
dedicated scale-matched runner. Importing it authorizes no CUDA, training,
evaluation, backward pass, optimizer construction, or optimizer step.
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
from contramamba import gen5_stage_e_learned_b_plane_afix_monly_control as afix


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "1ea8d156c86a7f8ec6b9ad638632e46b54b0de38"
)
ARM = "E-BFREE-AFIX-MONLY-SCALEMATCH"
TRAINING_SEEDS = afix.TRAINING_SEEDS
SOURCE_CORRECTIONS = afix.SOURCE_CORRECTIONS
EXPECTED_TRAINABLE_NUMEL = afix.EXPECTED_TRAINABLE_NUMEL


class LearnedBPlaneAFixMOnlyScaleMatchError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LearnedBPlaneAFixMOnlyScaleMatchError(message)


tensor_sha256 = afix.tensor_sha256
load_seed_matched_afix_monly_scalematch_source = (
    afix.load_seed_matched_afix_monly_source
)


class LearnedBPlaneAFixMOnlyScaleMatchCorrection(
    afix.LearnedBPlaneAFixMOnlyCorrection
):
    """Exact AFIX-MONLY parameterization with a distinct experiment identity."""

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
        require(
            not self.A_theta.weight.requires_grad,
            "SCALEMATCH_A_REQUIRES_GRAD",
        )
        require(
            self.M_theta.weight.requires_grad,
            "SCALEMATCH_M_NOT_TRAINABLE",
        )
        require(
            int(torch.count_nonzero(self.M_theta.weight).item()) == 0,
            "SCALEMATCH_M_NONZERO",
        )


def install_learned_b_plane_afix_monly_scalematch_layer22_wrapper(
    model: nn.Module,
    *,
    q_bfree: torch.Tensor,
    a_free: torch.Tensor,
    seed: int,
    shape: CorrectionShape = FROZEN_SHAPE,
    strict_frozen_dimensions: bool = True,
) -> StageELayer22MixerWrapper:
    before = parent_parameter_fingerprint(model)
    wrapper = afix.install_learned_b_plane_afix_monly_layer22_wrapper(
        model,
        q_bfree=q_bfree,
        a_free=a_free,
        seed=seed,
        shape=shape,
        strict_frozen_dimensions=strict_frozen_dimensions,
    )
    correction = wrapper.correction
    correction.arm = ARM
    require(
        tensor_sha256(correction.A_theta.weight)
        == tensor_sha256(
            a_free.detach()
            .cpu()
            .contiguous()
            .to(dtype=correction.A_theta.weight.dtype)
        ),
        "SCALEMATCH_INSTALL_A_IDENTITY",
    )
    require(
        not correction.A_theta.weight.requires_grad,
        "SCALEMATCH_INSTALL_A_REQUIRES_GRAD",
    )
    require(
        correction.M_theta.weight.requires_grad,
        "SCALEMATCH_INSTALL_M_NOT_TRAINABLE",
    )
    require(
        int(torch.count_nonzero(correction.M_theta.weight).item()) == 0,
        "SCALEMATCH_INSTALL_M_NONZERO",
    )
    require(
        parent_parameter_fingerprint(model) == before,
        "SCALEMATCH_PARENT_FINGERPRINT_CHANGED",
    )
    return wrapper


def zero_output_firewall(correction: nn.Module) -> dict[str, Any]:
    result = afix.zero_output_firewall(correction)
    require(
        not correction.A_theta.weight.requires_grad,
        "SCALEMATCH_FIREWALL_A_REQUIRES_GRAD",
    )
    require(
        correction.M_theta.weight.requires_grad,
        "SCALEMATCH_FIREWALL_M_NOT_TRAINABLE",
    )
    return {
        **result,
        "A_requires_grad": False,
        "M_requires_grad": True,
    }


def trainable_parameter_audit(correction: nn.Module) -> dict[str, Any]:
    result = afix.trainable_parameter_audit(correction)
    require(
        result["trainable_tensor_names"] == ["M_theta.weight"],
        "SCALEMATCH_TRAINABLE_NAMES",
    )
    require(
        result["trainable_tensor_count"] == 1,
        "SCALEMATCH_TRAINABLE_TENSOR_COUNT",
    )
    require(
        result["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL == 4,
        "SCALEMATCH_TRAINABLE_NUMEL",
    )
    return result
