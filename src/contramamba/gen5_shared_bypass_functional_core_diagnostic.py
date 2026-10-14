"""Gen5 shared-bypass functional-core *candidate* diagnostic companion.

Implementation-only, inference-only helper. Not an evaluation command or authority.
Do not modify historical Phase2/3A modules or load confirmatory data.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from contramamba.gen5_phase2_state_update_ownership import (
    accelerated_correction_mixer_contribution,
    validate_runtime_mixer_dimensions,
)

SEEDS = (6201, 6202, 6203)
ARMS = (
    "ZERO", "FULL", "SHARED_ONLY", "RESIDUAL_ONLY",
    "SHARED_NORMMATCH", "RESIDUAL_NORMMATCH",
)
TARGET_LAYER = 22
EXPECTED_SOURCE_TENSOR_SHA256 = {6201: {'A_theta.weight': '23796c2fbdaeee6e76d58bf613b85fdf52d25a47c86d98bc9376422478e79469',
        'B_theta.weight': '16d51fc2a579a61def0620542ea1a9c78ef13c2dcf35645b87a01626c83403ff'},
 6202: {'A_theta.weight': '46472cf48fd5973fc2635c99ebc3d0b77798141c359ae5798b1ec9019e1f24f9',
        'B_theta.weight': '7a9165f0db7baacdbeccc930ae3d50c5910948931366385007d88bd27f3ffab4'},
 6203: {'A_theta.weight': '5f47d66270a2ee2900741395e54dd274d4cb12e8359cbc223ea9fa1be07911c8',
        'B_theta.weight': 'f5d4b05bb91dcee119d7e9b1308ff60ca8a914f882854b503973977cc114b862'}}
# Values are frozen from SHA256-verified preflight JSON:
# 82f93e692a84c6890c36179fe522f1ee30e61ccf002f07836cfbe93f26a46641
FROZEN_ARM_COEFFICIENTS = {6201: {'FULL': (1.0, 0.0, 0.0),
        'RESIDUAL_NORMMATCH': (1.1712125995962777, -0.38524845132108737, -0.33084494355593846),
        'RESIDUAL_ONLY': (1.0, -0.32893127298484004, -0.28248069024358363),
        'SHARED_NORMMATCH': (0.0, 0.631861458445249, 0.5426320802526966),
        'SHARED_ONLY': (0.0, 0.32893127298484004, 0.28248069024358363),
        'ZERO': (0.0, 0.0, 0.0)},
 6202: {'FULL': (0.0, 1.0, 0.0),
        'RESIDUAL_NORMMATCH': (-0.3805733794729952, 1.1926311141736292, -0.39648829569104765),
        'RESIDUAL_ONLY': (-0.3191040171182297, 1.0, -0.3324483916099852),
        'SHARED_NORMMATCH': (0.5855883151114114, 0.0, 0.6100765990428153),
        'SHARED_ONLY': (0.3191040171182297, 0.0, 0.3324483916099852),
        'ZERO': (0.0, 0.0, 0.0)},
 6203: {'FULL': (0.0, 0.0, 1.0),
        'RESIDUAL_NORMMATCH': (-0.31938378758410957, -0.3874549660407846, 1.1743163887116013),
        'RESIDUAL_ONLY': (-0.2719742231772145, -0.3299408658222679, 1.0),
        'SHARED_NORMMATCH': (0.5187789994143587, 0.6293478486218665, 0.0),
        'SHARED_ONLY': (0.2719742231772145, 0.3299408658222679, 0.0),
        'ZERO': (0.0, 0.0, 0.0)}}


class DiagnosticContractError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise DiagnosticContractError(reason)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def validate_frozen_sources(source_corrections: Mapping[int, nn.Module]) -> None:
    """Authenticate three frozen rank-2 C0 corrections by exact tensor bytes."""
    _require(set(source_corrections) == set(SEEDS), "SOURCE_SEED_SET_MISMATCH")
    for seed in SEEDS:
        correction = source_corrections[seed]
        _require(getattr(correction, "arm", None) == "G5-C0", f"ARM_NOT_C0:{seed}")
        a = correction.A_theta.weight
        b = correction.B_theta.weight
        _require(tuple(a.shape) == (2, 768), f"SOURCE_A_SHAPE:{seed}")
        _require(tuple(b.shape) == (24576, 2), f"SOURCE_B_SHAPE:{seed}")
        _require(a.dtype == torch.float32 and b.dtype == torch.float32,
                 f"SOURCE_DTYPE:{seed}")
        _require(not a.requires_grad and not b.requires_grad, f"SOURCE_TRAINABLE:{seed}")
        _require(not correction.training, f"SOURCE_TRAINING_MODE:{seed}")
        _require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                 f"SOURCE_NONFINITE:{seed}")
        expect = EXPECTED_SOURCE_TENSOR_SHA256[seed]
        _require(_tensor_sha256(a) == expect["A_theta.weight"],
                 f"SOURCE_A_IDENTITY:{seed}")
        _require(_tensor_sha256(b) == expect["B_theta.weight"],
                 f"SOURCE_B_IDENTITY:{seed}")


def select_frozen_coefficients(heldout_seed: int, arm: str) -> tuple[float, float, float]:
    _require(type(heldout_seed) is int and heldout_seed in SEEDS,
             "UNAUTHORIZED_HELDOUT_SEED")
    _require(arm in ARMS, "UNAUTHORIZED_ARM")
    coeffs = FROZEN_ARM_COEFFICIENTS[heldout_seed][arm]
    _require(len(coeffs) == 3 and all(math.isfinite(c) for c in coeffs),
             "INVALID_FROZEN_COEFFICIENTS")
    return coeffs


class FrozenSharedBypassDiagnosticMixer(nn.Module):
    """Preserve historical native mixer call; sum frozen C0 correction outputs.

    No training, no checkpoint mutation, no native mixer changes. Each donor
    calls the *existing* rank-2 accelerated correction path at the original
    WRITE22 location. The new module does NOT implement a new recurrence.

    This is a candidate for subsequent static/synthetic contract verification;
    scientific model forward/evaluation remains separately unauthorized.
    """

    def __init__(
        self,
        *,
        native_mixer: nn.Module,
        source_corrections: Mapping[int, nn.Module],
        heldout_seed: int,
        arm: str,
    ) -> None:
        super().__init__()
        validate_runtime_mixer_dimensions(native_mixer)
        _require(int(getattr(native_mixer, "layer_idx", -1)) == TARGET_LAYER,
                 "NATIVE_LAYER_MISMATCH")
        _require(not native_mixer.training, "NATIVE_MIXER_NOT_EVAL")
        validate_frozen_sources(source_corrections)
        self.coefficients = select_frozen_coefficients(heldout_seed, arm)
        self.native_mixer = native_mixer
        self.correction = nn.ModuleDict(
            {str(seed): source_corrections[seed] for seed in SEEDS}
        )
        self.layer_idx = TARGET_LAYER
        self.heldout_seed = heldout_seed
        self.arm = arm
        self._phase2_active_mask: torch.Tensor | None = None

    @contextmanager
    def active_mask(self, attention_mask: torch.Tensor) -> Iterator[None]:
        _require(self._phase2_active_mask is None, "ACTIVE_MASK_REENTRANT")
        _require(attention_mask.ndim == 2, "ACTIVE_MASK_RANK")
        self._phase2_active_mask = attention_mask
        try:
            yield
        finally:
            self._phase2_active_mask = None

    def train(self, mode: bool = True) -> "FrozenSharedBypassDiagnosticMixer":
        _require(not mode, "DIAGNOSTIC_TRAINING_FORBIDDEN")
        super().train(False)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require(not torch.is_grad_enabled(), "GRAD_ENABLED_FORBIDDEN")
        _require(not self.training and not self.native_mixer.training,
                 "EVAL_ONLY")
        _require(cache_params is None and cache_position is None,
                 "CACHE_FORBIDDEN")
        mask = attention_mask if attention_mask is not None else self._phase2_active_mask
        _require(mask is not None, "ACTIVE_MASK_REQUIRED")
        _require(hidden_states.ndim == 3 and hidden_states.shape[-1] == 768,
                 "INPUT_SHAPE")
        _require(tuple(mask.shape) == tuple(hidden_states.shape[:2]),
                 "ACTIVE_MASK_SHAPE")
        _require(all(not p.requires_grad for p in self.native_mixer.parameters()),
                 "NATIVE_PARAMETERS_NOT_FROZEN")
        # Preserve original frozen native call exactly, including original
        # argument attention_mask (not the out-of-band effective mask).
        native = self.native_mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        total_correction: torch.Tensor | None = None
        for seed, coeff in zip(SEEDS, self.coefficients):
            if coeff == 0.0:
                continue
            contribution = accelerated_correction_mixer_contribution(
                self.native_mixer,
                hidden_states,
                self.correction[str(seed)],
                attention_mask=mask,
                checkpoint_recompute=False,
            )
            _require(tuple(contribution.shape) == tuple(native.shape),
                     "CORRECTION_OUTPUT_SHAPE")
            term = contribution * coeff
            total_correction = term if total_correction is None else total_correction + term
        if total_correction is None:
            return native
        return native + total_correction
