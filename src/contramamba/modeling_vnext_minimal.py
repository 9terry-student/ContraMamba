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
VALID_VNEXT_ROUTER_MODES = {
    "learned_only",
    "product",
    "min",
    "learned_x_product",
    "learned_x_sufficiency",
    "sufficiency_only",
    "learned_x_frame_sufficiency",
    "learned_x_predicate_sufficiency",
}
VALID_SLOT_MISMATCH_INPUT_MODES = {
    "sufficiency_repr",
    "channel_concat",
    "pooled_pair_concat",
    "pooled_pair_absdiff_product",
}
VALID_SLOT_MISMATCH_HEAD_TYPES = {"linear", "mlp"}


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
        vnext_enable_segmented_dual_pass: bool = False,
        vnext_segmented_context_role: str = "diagnostic_only",
        vnext_context_risk_cap_alpha: float = 0.0,
        vnext_context_risk_threshold: float = 0.5,
        vnext_context_risk_source: str = "context_not_entitled_prob",
        vnext_use_slot_mismatch_head: bool = False,
        vnext_slot_mismatch_detach_input: bool = True,
        vnext_slot_mismatch_input_mode: str = "sufficiency_repr",
        vnext_slot_mismatch_head_type: str = "linear",
    ) -> None:
        super().__init__()
        if vnext_router_mode not in VALID_VNEXT_ROUTER_MODES:
            raise ValueError(
                f"unsupported vnext_router_mode: {vnext_router_mode!r}; "
                f"expected one of {sorted(VALID_VNEXT_ROUTER_MODES)}"
            )
        if vnext_slot_mismatch_input_mode not in VALID_SLOT_MISMATCH_INPUT_MODES:
            raise ValueError(
                "unsupported vnext_slot_mismatch_input_mode: "
                f"{vnext_slot_mismatch_input_mode!r}; expected one of "
                f"{sorted(VALID_SLOT_MISMATCH_INPUT_MODES)}"
            )
        if vnext_slot_mismatch_head_type not in VALID_SLOT_MISMATCH_HEAD_TYPES:
            raise ValueError(
                "unsupported vnext_slot_mismatch_head_type: "
                f"{vnext_slot_mismatch_head_type!r}; expected one of "
                f"{sorted(VALID_SLOT_MISMATCH_HEAD_TYPES)}"
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
        self.vnext_enable_segmented_dual_pass = bool(vnext_enable_segmented_dual_pass)
        self.vnext_segmented_context_role = str(vnext_segmented_context_role)
        self.vnext_context_risk_cap_alpha = float(vnext_context_risk_cap_alpha)
        self.vnext_context_risk_threshold = float(vnext_context_risk_threshold)
        self.vnext_context_risk_source = str(vnext_context_risk_source)
        self.vnext_use_slot_mismatch_head = bool(vnext_use_slot_mismatch_head)
        self.vnext_slot_mismatch_detach_input = bool(vnext_slot_mismatch_detach_input)
        self.vnext_slot_mismatch_input_mode = str(vnext_slot_mismatch_input_mode)
        self.vnext_slot_mismatch_head_type = str(vnext_slot_mismatch_head_type)

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
        self.slot_mismatch_head_input_dim = self._slot_mismatch_input_dim(
            input_mode=self.vnext_slot_mismatch_input_mode,
            frame_size=frame_size,
            predicate_size=predicate_size,
            sufficiency_size=sufficiency_size,
        )
        self.slot_mismatch_head = (
            self._build_slot_mismatch_head(
                input_dim=self.slot_mismatch_head_input_dim,
                head_type=self.vnext_slot_mismatch_head_type,
                dropout=dropout,
            )
            if self.vnext_use_slot_mismatch_head
            else None
        )
        self.not_entitled_bias = nn.Parameter(torch.tensor(float(not_entitled_bias_init)))
        self.raw_not_entitled_alpha = nn.Parameter(
            torch.tensor(_inverse_softplus(not_entitled_alpha_init))
        )

    @staticmethod
    def _slot_mismatch_input_dim(
        *,
        input_mode: str,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
    ) -> int:
        if input_mode == "sufficiency_repr":
            return sufficiency_size
        if input_mode == "channel_concat":
            return frame_size + predicate_size + sufficiency_size
        if input_mode == "pooled_pair_concat":
            return frame_size * 2
        if input_mode == "pooled_pair_absdiff_product":
            return frame_size * 4
        raise ValueError(f"unsupported slot mismatch input_mode: {input_mode!r}")

    @staticmethod
    def _build_slot_mismatch_head(
        *,
        input_dim: int,
        head_type: str,
        dropout: float,
    ) -> nn.Module:
        if head_type == "linear":
            return nn.Linear(input_dim, 1)
        if head_type == "mlp":
            hidden_dim = max(32, min(128, input_dim))
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        raise ValueError(f"unsupported slot mismatch head_type: {head_type!r}")

    def _slot_mismatch_features(
        self,
        *,
        frame: dict[str, torch.Tensor | None],
        predicate: dict[str, torch.Tensor | None],
        sufficiency: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.vnext_slot_mismatch_input_mode == "sufficiency_repr":
            return sufficiency["sufficiency_repr"]
        if self.vnext_slot_mismatch_input_mode == "channel_concat":
            return torch.cat(
                [
                    frame["frame_pair_repr"],
                    predicate["predicate_pair_repr"],
                    sufficiency["sufficiency_repr"],
                ],
                dim=-1,
            )
        claim_pooled = frame["claim_frame_state"]
        evidence_pooled = frame["evidence_frame_state"]
        if self.vnext_slot_mismatch_input_mode == "pooled_pair_concat":
            return torch.cat([claim_pooled, evidence_pooled], dim=-1)
        if self.vnext_slot_mismatch_input_mode == "pooled_pair_absdiff_product":
            return torch.cat(
                [
                    claim_pooled,
                    evidence_pooled,
                    torch.abs(claim_pooled - evidence_pooled),
                    claim_pooled * evidence_pooled,
                ],
                dim=-1,
            )
        raise ValueError(
            "unsupported slot mismatch input_mode: "
            f"{self.vnext_slot_mismatch_input_mode!r}"
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
        if vnext_router_mode == "learned_x_sufficiency":
            return learned_entitlement_prob * sufficiency_prob
        if vnext_router_mode == "sufficiency_only":
            return sufficiency_prob
        if vnext_router_mode == "learned_x_frame_sufficiency":
            return learned_entitlement_prob * frame_prob * sufficiency_prob
        if vnext_router_mode == "learned_x_predicate_sufficiency":
            return learned_entitlement_prob * predicate_coverage_prob * sufficiency_prob
        raise ValueError(f"unsupported vnext_router_mode: {vnext_router_mode!r}")

    def _compute_vnext_logits_from_states(
        self,
        *,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> dict[str, Any]:
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
        slot_mismatch_logit = None
        slot_mismatch_prob = None
        if self.slot_mismatch_head is not None:
            slot_mismatch_repr = self._slot_mismatch_features(
                frame=frame,
                predicate=predicate,
                sufficiency=sufficiency,
            )
            if self.vnext_slot_mismatch_detach_input:
                slot_mismatch_repr = slot_mismatch_repr.detach()
            slot_mismatch_logit = self.slot_mismatch_head(slot_mismatch_repr).squeeze(-1)
            slot_mismatch_prob = torch.sigmoid(slot_mismatch_logit)
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
        return {
            "logits": final_logits,
            "support_score": support_score,
            "refute_score": refute_score,
            "ne_score": ne_score,
            "entitlement_for_decision": entitlement_for_decision,
            "compositional_entitlement_prob": compositional_entitlement_prob,
            "learned_entitlement_logit": learned_entitlement_logit,
            "learned_entitlement_prob": learned_entitlement_prob,
            "slot_mismatch_logit": slot_mismatch_logit,
            "slot_mismatch_prob": slot_mismatch_prob,
            "frame": frame,
            "predicate": predicate,
            "sufficiency": sufficiency,
            "polarity": polarity,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        core_input_ids: torch.Tensor | None = None,
        core_attention_mask: torch.Tensor | None = None,
        core_claim_mask: torch.Tensor | None = None,
        core_evidence_mask: torch.Tensor | None = None,
        context_input_ids: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        context_claim_mask: torch.Tensor | None = None,
        context_evidence_mask: torch.Tensor | None = None,
        context_empty: torch.Tensor | None = None,
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

        segmented_active = self.vnext_enable_segmented_dual_pass and core_input_ids is not None
        if segmented_active:
            input_ids = core_input_ids
            attention_mask = core_attention_mask if core_attention_mask is not None else attention_mask
            claim_mask = core_claim_mask if core_claim_mask is not None else claim_mask
            evidence_mask = core_evidence_mask if core_evidence_mask is not None else evidence_mask
            encoder_hidden_states = None

        if encoder_hidden_states is None:
            backbone_outputs = self.mamba(input_ids=input_ids)
            token_states = backbone_outputs.last_hidden_state
        else:
            if encoder_hidden_states.shape[:2] != input_ids.shape:
                raise ValueError(
                    "encoder_hidden_states must match input_ids batch/sequence dimensions"
                )
            token_states = encoder_hidden_states

        context_token_states = None
        context_attention_for_decision = None
        context_claim_mask_for_decision = None
        context_evidence_mask_for_decision = None
        context_empty_bool = None
        context_rep_norm = None
        core_context_cosine = None
        if segmented_active and context_input_ids is not None:
            context_outputs = self.mamba(input_ids=context_input_ids)
            context_token_states = context_outputs.last_hidden_state
            context_attention = (
                context_attention_mask
                if context_attention_mask is not None
                else torch.ones_like(context_input_ids, dtype=torch.bool)
            )
            context_mask = (
                context_evidence_mask
                if context_evidence_mask is not None
                else context_attention
            )
            context_attention_for_decision = context_attention
            context_claim_mask_for_decision = (
                context_claim_mask
                if context_claim_mask is not None
                else torch.zeros_like(context_attention)
            )
            context_evidence_mask_for_decision = context_mask
            context_weights = (
                context_mask.float()
                / context_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            context_rep = (context_token_states * context_weights.unsqueeze(-1)).sum(dim=1)
            if context_empty is not None:
                context_empty_bool = context_empty.bool()
                context_rep = context_rep.masked_fill(context_empty_bool.unsqueeze(-1), 0.0)
            core_weights = (
                evidence_mask.float()
                / evidence_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            core_rep_for_diag = (token_states * core_weights.unsqueeze(-1)).sum(dim=1)
            context_rep_norm = context_rep.norm(dim=-1)
            core_context_cosine = F.cosine_similarity(core_rep_for_diag, context_rep, dim=-1)
            if context_empty_bool is not None:
                core_context_cosine = core_context_cosine.masked_fill(context_empty_bool, 0.0)

        core_logits = self._compute_vnext_logits_from_states(
            token_states=token_states,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
        )
        frame = core_logits["frame"]
        predicate = core_logits["predicate"]
        sufficiency = core_logits["sufficiency"]
        polarity = core_logits["polarity"]
        learned_entitlement_logit = core_logits["learned_entitlement_logit"]
        learned_entitlement_prob = core_logits["learned_entitlement_prob"]
        slot_mismatch_logit = core_logits["slot_mismatch_logit"]
        slot_mismatch_prob = core_logits["slot_mismatch_prob"]
        compositional_entitlement_prob = core_logits["compositional_entitlement_prob"]
        entitlement_for_decision = core_logits["entitlement_for_decision"]
        support_score = core_logits["support_score"]
        refute_score = core_logits["refute_score"]
        ne_score = core_logits["ne_score"]
        final_logits = core_logits["logits"]
        base_logits = final_logits
        logits_before_context_cap = final_logits
        prediction_before_context_cap = final_logits.argmax(dim=-1)
        context_only_logits = None
        context_only_prediction = None
        context_risk = None
        context_risk_excess = None
        context_cap_factor = None
        context_cap_applied = None
        context_cap_notes = "context_cap_inactive"
        context_risk_cap_active = (
            segmented_active
            and self.vnext_segmented_context_role == "risk_cap"
            and context_token_states is not None
        )
        if context_risk_cap_active:
            context_logits = self._compute_vnext_logits_from_states(
                token_states=context_token_states,
                attention_mask=context_attention_for_decision,
                claim_mask=context_claim_mask_for_decision,
                evidence_mask=context_evidence_mask_for_decision,
            )
            context_only_logits = context_logits["logits"]
            context_only_prediction = context_only_logits.argmax(dim=-1)
            context_probs = torch.softmax(context_only_logits, dim=-1)
            if self.vnext_context_risk_source == "context_not_entitled_prob":
                context_risk = context_probs[:, NOT_ENTITLED_ID]
                context_cap_notes = "risk_source=context_not_entitled_prob"
            elif self.vnext_context_risk_source == "context_uncertainty":
                context_risk = 1.0 - context_probs.max(dim=-1).values
                context_cap_notes = "risk_source=context_uncertainty"
            else:
                context_risk = torch.zeros_like(context_probs[:, NOT_ENTITLED_ID])
                context_cap_notes = "risk_source_unknown_zero_fallback"
            context_cap_notes = context_cap_notes + ";context_only_logits=computed"
            if context_empty_bool is not None:
                context_risk = context_risk.masked_fill(context_empty_bool, 0.0)
            threshold = float(self.vnext_context_risk_threshold)
            alpha = float(self.vnext_context_risk_cap_alpha)
            denom = max(1e-6, 1.0 - threshold)
            context_risk_excess = ((context_risk - threshold).clamp_min(0.0) / denom)
            context_cap_factor = (1.0 - alpha * context_risk_excess).clamp(0.0, 1.0)
            if context_empty_bool is not None:
                context_cap_factor = context_cap_factor.masked_fill(context_empty_bool, 1.0)
            context_cap_applied = context_cap_factor < 0.999999
            if alpha > 0.0:
                capped_logits = final_logits.clone()
                log_cap = torch.log(context_cap_factor.clamp_min(1e-6))
                capped_logits[:, REFUTE_ID] = capped_logits[:, REFUTE_ID] + log_cap
                capped_logits[:, SUPPORT_ID] = capped_logits[:, SUPPORT_ID] + log_cap
                final_logits = capped_logits
                context_cap_notes = context_cap_notes + ";symmetric_refute_support_log_cap"
            elif self.vnext_segmented_context_role == "risk_cap" and alpha == 0.0:
                context_cap_notes = context_cap_notes + ";alpha_zero_noop"
            elif self.vnext_segmented_context_role != "risk_cap":
                context_cap_notes = context_cap_notes + ";role_noop"
        logits_after_context_cap = final_logits
        prediction_after_context_cap = final_logits.argmax(dim=-1)

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
            "slot_mismatch_logit": slot_mismatch_logit,
            "slot_mismatch_prob": slot_mismatch_prob,
            "vnext_use_slot_mismatch_head": self.vnext_use_slot_mismatch_head,
            "vnext_slot_mismatch_detach_input": self.vnext_slot_mismatch_detach_input,
            "vnext_slot_mismatch_input_mode": self.vnext_slot_mismatch_input_mode,
            "vnext_slot_mismatch_head_type": self.vnext_slot_mismatch_head_type,
            "slot_mismatch_head_input_dim": self.slot_mismatch_head_input_dim,
            "vnext_router_mode": self.vnext_router_mode,
            "vnext_final_logit_order": FINAL_LOGIT_ORDER,
            "vnext_segmented_dual_pass_active": segmented_active,
            "vnext_primary_rep_source": "core_rep" if segmented_active else "single_pass",
            "vnext_core_rep_norm": sufficiency["sufficiency_repr"].norm(dim=-1)
            if segmented_active
            else None,
            "vnext_context_rep_norm": context_rep_norm,
            "vnext_core_context_cosine": core_context_cosine,
            "vnext_context_risk_cap_active": context_risk_cap_active,
            "vnext_context_risk_source": self.vnext_context_risk_source,
            "vnext_context_risk": context_risk,
            "vnext_context_risk_threshold": self.vnext_context_risk_threshold,
            "vnext_context_risk_cap_alpha": self.vnext_context_risk_cap_alpha,
            "vnext_context_risk_excess": context_risk_excess,
            "vnext_context_cap_factor": context_cap_factor,
            "vnext_context_cap_applied": context_cap_applied,
            "vnext_logits_before_context_cap": logits_before_context_cap,
            "vnext_logits_after_context_cap": logits_after_context_cap,
            "vnext_prediction_before_context_cap": prediction_before_context_cap,
            "vnext_prediction_after_context_cap": prediction_after_context_cap,
            "vnext_context_only_logits": context_only_logits,
            "vnext_context_only_prediction": context_only_prediction,
            "vnext_context_cap_notes": context_cap_notes,
            "not_entitled_alpha": self.not_entitled_alpha(),
            **frame,
            **predicate,
            **sufficiency,
            **polarity,
            "token_states": token_states if return_token_states else None,
            "loss": total_loss,
            **losses,
        }
