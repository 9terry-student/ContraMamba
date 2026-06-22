from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("alpha_init must be positive")
    return math.log(math.expm1(value))


class FinalEntitlementDecisionHead(nn.Module):
    """Final authority for REFUTE / NOT_ENTITLED / SUPPORT."""

    VALID_DECISION_MODES = {"explicit_product", "logit_sum"}

    def __init__(
        self,
        not_entitled_bias_init: float = 0.0,
        alpha_init: float = 1.0,
        decision_mode: str = "explicit_product",
    ) -> None:
        super().__init__()
        if decision_mode not in self.VALID_DECISION_MODES:
            raise ValueError(f"unsupported decision_mode: {decision_mode}")
        self.decision_mode = decision_mode
        self.not_entitled_bias = nn.Parameter(
            torch.tensor(float(not_entitled_bias_init))
        )
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(alpha_init)))

    def forward(
        self,
        frame_prob: torch.Tensor,
        predicate_coverage_prob: torch.Tensor,
        sufficiency_prob: torch.Tensor,
        positive_energy: torch.Tensor,
        negative_energy: torch.Tensor,
        decision_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = decision_mode or self.decision_mode
        if mode not in self.VALID_DECISION_MODES:
            raise ValueError(f"unsupported decision_mode: {mode}")

        if mode == "explicit_product":
            entitlement_prob = (
                frame_prob * predicate_coverage_prob * sufficiency_prob
            )
        else:
            eps = torch.finfo(frame_prob.dtype).eps
            gate_probs = (frame_prob, predicate_coverage_prob, sufficiency_prob)
            gate_logits = [
                torch.logit(prob.clamp(min=eps, max=1.0 - eps))
                for prob in gate_probs
            ]
            entitlement_prob = torch.sigmoid(torch.stack(gate_logits).sum(dim=0))

        support_logit = entitlement_prob * positive_energy
        refute_logit = entitlement_prob * negative_energy
        alpha = F.softplus(self.raw_alpha)
        not_entitled_logit = self.not_entitled_bias + alpha * (
            1.0 - entitlement_prob
        )
        logits = torch.stack(
            [refute_logit, not_entitled_logit, support_logit], dim=-1
        )
        return {
            "entitlement_prob": entitlement_prob,
            "support_logit": support_logit,
            "refute_logit": refute_logit,
            "not_entitled_logit": not_entitled_logit,
            "logits": logits,
        }

