"""ContraMamba-vNext-minimal: entitlement-first state-space controller.

Canonical final label order is preserved throughout this file:
REFUTE=0, NOT_ENTITLED=1, SUPPORT=2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .heads import FrameGate, PolarityEnergyHead, PredicateCoverageHead, SufficiencyGate


FINAL_LOGIT_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
REFUTE_ID = 0
NOT_ENTITLED_ID = 1
SUPPORT_ID = 2
VALID_VNEXT_ROUTER_MODES = {"learned_only", "product", "min", "learned_x_product"}


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("value must be positive")
    return math.log(math.expm1(value))


class ContraMambaVNextMinimal(nn.Module):
    """Minimal vNext model with an explicit entitlement-first final router.

    The model reuses the stable v5/v6B slot heads and Mamba backbone pattern, but
    owns a clean final decision surface. SUPPORT and REFUTE are gated by
    ``entitlement_for_decision``; NOT_ENTITLED is an explicit low-entitlement
    score rather than the residual of low polarity.
    """

    def __init__(
        self,
        model_name: str = "state-spaces/mamba-130m-hf",
        frame_size: int = 256,
        predicate_size: int = 256,
        sufficiency_size: int = 256,
        energy_size: int = 64,
        dropout: float = 0.1,
        freeze_a_log: bool = True,
        return_token_diagnostics: bool = False,
        vnext_router_mode: str = "learned_x_product",
        not_entitled_bias_init: float = 0.0,
        not_entitled_alpha_init: float = 1.0,
        backbone: nn.Module | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if vnext_router_mode not in VALID_VNEXT_ROUTER_MODES:
            raise ValueError(
                f"unsupported vnext_router_mode: {vnext_router_mode!r}; "
                f"expected one of {sorted(VALID_VNEXT_ROUTER_MODES)}"
            )

        if backbone is None:
            try:
                from transformers import MambaConfig, MambaModel
            except ImportError as exc:
                raise ImportError(
                    "transformers is required when no backbone is supplied"
                ) from exc
            config = MambaConfig.from_pretrained(model_name)
            config.use_mamba_kernels = True
            backbone = MambaModel.from_pretrained(model_name, config=config)

        self.mamba = backbone
        inferred_hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        hidden_size = hidden_size or inferred_hidden_size
        if hidden_size is None:
            raise ValueError("hidden_size is required when the backbone has no config")

        if freeze_a_log:
            for name, parameter in self.mamba.named_parameters():
                if "A_log" in name:
                    parameter.requires_grad = False

        self.return_token_diagnostics = return_token_diagnostics
        self.vnext_router_mode = vnext_router_mode

        self.frame_gate = FrameGate(
            hidden_size, frame_size, dropout, return_token_diagnostics
        )
        self.predicate_coverage_head = PredicateCoverageHead(
            hidden_size, frame_size, predicate_size, dropout, return_token_diagnostics
        )
        self.sufficiency_gate = SufficiencyGate(
            frame_size, predicate_size, sufficiency_size, dropout
        )
        self.polarity_energy_head = PolarityEnergyHead(
            frame_size, predicate_size, sufficiency_size, energy_size, dropout
        )
        self.learned_entitlement_head = nn.Linear(sufficiency_size, 1)
        self.not_entitled_bias = nn.Parameter(torch.tensor(float(not_entitled_bias_init)))
        self.raw_not_entitled_alpha = nn.Parameter(
            torch.tensor(_inverse_softplus(not_entitled_alpha_init))
        )

    def not_entitled_alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_not_entitled_alpha)

    def _compose_entitlement(
        self,
        *,
        learned_entitlement_prob: torch.Tensor,
        compositional_entitlement_prob: torch.Tensor,
        frame_prob: torch.Tensor,
        predicate_coverage_prob: torch.Tensor,
        sufficiency_prob: torch.Tensor,
        vnext_router_mode: str,
    ) -> torch.Tensor:
        if vnext_router_mode == "learned_only":
            return learned_entitlement_prob
        if vnext_router_mode == "product":
            return compositional_entitlement_prob
        if vnext_router_mode == "min":
            return torch.stack(
                [
                    learned_entitlement_prob,
                    frame_prob,
                    predicate_coverage_prob,
                    sufficiency_prob,
                ],
                dim=0,
            ).amin(dim=0)
        if vnext_router_mode == "learned_x_product":
            return learned_entitlement_prob * compositional_entitlement_prob
        raise ValueError(f"unsupported vnext_router_mode: {vnext_router_mode!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        final_labels: torch.Tensor | None = None,
        frame_compatible_labels: torch.Tensor | None = None,
        predicate_covered_labels: torch.Tensor | None = None,
        sufficiency_labels: torch.Tensor | None = None,
        polarity_labels: torch.Tensor | None = None,
        polarity_mask: torch.Tensor | None = None,
        intervention_types: torch.Tensor | None = None,
        pair_ids: torch.Tensor | None = None,
        return_token_states: bool = False,
        decision_mode: str | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        temporal_mismatch_flags: torch.Tensor | None = None,
        predicate_mismatch_flags: torch.Tensor | None = None,
        temporal_adapter_final_penalty_scale: float = 0.0,
        temporal_channel_gated_penalty_scale: float = 0.0,
    ) -> dict[str, Any]:
        del (
            intervention_types,
            pair_ids,
            decision_mode,
            temporal_mismatch_flags,
            predicate_mismatch_flags,
            temporal_adapter_final_penalty_scale,
            temporal_channel_gated_penalty_scale,
        )

        if encoder_hidden_states is None:
            backbone_outputs = self.mamba(input_ids=input_ids)
            token_states = backbone_outputs.last_hidden_state
        else:
            if encoder_hidden_states.shape[:2] != input_ids.shape:
                raise ValueError(
                    "encoder_hidden_states must match input_ids batch/sequence dimensions"
                )
            token_states = encoder_hidden_states

        frame = self.frame_gate(token_states, attention_mask, claim_mask, evidence_mask)
        predicate = self.predicate_coverage_head(
            token_states=token_states,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
            claim_frame_state=frame["claim_frame_state"],
            evidence_frame_state=frame["evidence_frame_state"],
            frame_pair_repr=frame["frame_pair_repr"],
            frame_prob=frame["frame_prob"],
        )
        sufficiency = self.sufficiency_gate(
            frame_pair_repr=frame["frame_pair_repr"],
            predicate_pair_repr=predicate["predicate_pair_repr"],
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
        )
        polarity = self.polarity_energy_head(
            frame_pair_repr=frame["frame_pair_repr"],
            predicate_pair_repr=predicate["predicate_pair_repr"],
            sufficiency_repr=sufficiency["sufficiency_repr"],
        )

        learned_entitlement_logit = self.learned_entitlement_head(
            sufficiency["sufficiency_repr"]
        ).squeeze(-1)
        learned_entitlement_prob = torch.sigmoid(learned_entitlement_logit)
        compositional_entitlement_prob = (
            frame["frame_prob"]
            * predicate["predicate_coverage_prob"]
            * sufficiency["sufficiency_prob"]
        )
        entitlement_for_decision = self._compose_entitlement(
            learned_entitlement_prob=learned_entitlement_prob,
            compositional_entitlement_prob=compositional_entitlement_prob,
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
            sufficiency_prob=sufficiency["sufficiency_prob"],
            vnext_router_mode=self.vnext_router_mode,
        )

        support_score = entitlement_for_decision * polarity["positive_energy"]
        refute_score = entitlement_for_decision * polarity["negative_energy"]
        ne_score = self.not_entitled_bias + self.not_entitled_alpha() * (
            1.0 - entitlement_for_decision
        )
        final_logits = torch.stack([refute_score, ne_score, support_score], dim=-1)
        base_logits = final_logits

        losses: dict[str, torch.Tensor | None] = {
            "label_loss": None,
            "frame_loss": None,
            "predicate_loss": None,
            "sufficiency_loss": None,
            "polarity_loss": None,
        }
        if final_labels is not None:
            losses["label_loss"] = F.cross_entropy(final_logits, final_labels)
        if frame_compatible_labels is not None:
            losses["frame_loss"] = F.binary_cross_entropy_with_logits(
                frame["frame_logit"], frame_compatible_labels.float()
            )
        if predicate_covered_labels is not None:
            losses["predicate_loss"] = F.binary_cross_entropy_with_logits(
                predicate["predicate_coverage_logit"],
                predicate_covered_labels.float(),
            )
        if sufficiency_labels is not None:
            losses["sufficiency_loss"] = F.binary_cross_entropy_with_logits(
                sufficiency["sufficiency_logit"], sufficiency_labels.float()
            )
        if polarity_labels is not None:
            if polarity_mask is None:
                polarity_mask = (
                    final_labels != NOT_ENTITLED_ID
                    if final_labels is not None
                    else polarity_labels != 0
                )
            active = polarity_mask.bool()
            if torch.any(active):
                polarity_logits = torch.stack(
                    [
                        torch.zeros_like(polarity["negative_energy"]),
                        polarity["negative_energy"],
                        polarity["positive_energy"],
                    ],
                    dim=-1,
                )
                losses["polarity_loss"] = F.cross_entropy(
                    polarity_logits[active], polarity_labels[active]
                )

        active_losses = [loss for loss in losses.values() if loss is not None]
        total_loss = torch.stack(active_losses).sum() if active_losses else None

        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "predictions": final_logits.argmax(dim=-1),
            "entitlement_prob": entitlement_for_decision,
            "support_logit": support_score,
            "refute_logit": refute_score,
            "not_entitled_logit": ne_score,
            "positive_energy": polarity["positive_energy"],
            "negative_energy": polarity["negative_energy"],
            "polarity_margin": polarity["polarity_margin"],
            "entitlement_for_decision": entitlement_for_decision,
            "compositional_entitlement_prob": compositional_entitlement_prob,
            "learned_entitlement_logit": learned_entitlement_logit,
            "learned_entitlement_prob": learned_entitlement_prob,
            "vnext_router_mode": self.vnext_router_mode,
            "vnext_final_logit_order": FINAL_LOGIT_ORDER,
            "not_entitled_alpha": self.not_entitled_alpha(),
            **frame,
            **predicate,
            **sufficiency,
            **polarity,
            "token_states": token_states if return_token_states else None,
            "loss": total_loss,
            **losses,
        }
