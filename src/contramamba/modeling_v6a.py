from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .heads import (
    CalibratedComposer,
    FinalEntitlementDecisionHead,
    FrameGate,
    PolarityEnergyHead,
    PredicateCoverageHead,
    SufficiencyGate,
)


class ContraMambaV6A(nn.Module):
    """Minimal v5-compatible ContraMamba-v6A model.

    v6A preserves the v5 gate and polarity heads, keeps the explicit-product
    final decision as a diagnostic path, and uses a learned calibrated composer
    as the final classifier authority.
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
        decision_mode: str = "explicit_product",
        backbone: nn.Module | None = None,
        hidden_size: int | None = None,
        composer_hidden_size: int = 64,
        product_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
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
        self.product_loss_weight = product_loss_weight
        self.frame_gate = FrameGate(
            hidden_size,
            frame_size,
            dropout,
            return_token_diagnostics,
        )
        self.predicate_coverage_head = PredicateCoverageHead(
            hidden_size,
            frame_size,
            predicate_size,
            dropout,
            return_token_diagnostics,
        )
        self.sufficiency_gate = SufficiencyGate(
            frame_size, predicate_size, sufficiency_size, dropout
        )
        self.polarity_energy_head = PolarityEnergyHead(
            frame_size,
            predicate_size,
            sufficiency_size,
            energy_size,
            dropout,
        )
        self.product_decision_head = FinalEntitlementDecisionHead(
            decision_mode=decision_mode
        )
        self.decision_head = self.product_decision_head
        self.composer = CalibratedComposer(
            hidden_size=composer_hidden_size,
            dropout=dropout,
        )

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
    ) -> dict[str, Any]:
        del intervention_types, pair_ids
        if encoder_hidden_states is None:
            backbone_outputs = self.mamba(input_ids=input_ids)
            token_states = backbone_outputs.last_hidden_state
        else:
            if encoder_hidden_states.shape[:2] != input_ids.shape:
                raise ValueError(
                    "encoder_hidden_states must match input_ids batch/sequence dimensions"
                )
            token_states = encoder_hidden_states

        frame = self.frame_gate(
            token_states, attention_mask, claim_mask, evidence_mask
        )
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
        product = self.product_decision_head(
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
            sufficiency_prob=sufficiency["sufficiency_prob"],
            positive_energy=polarity["positive_energy"],
            negative_energy=polarity["negative_energy"],
            decision_mode=decision_mode,
        )
        composer = self.composer(
            frame_logit=frame["frame_logit"],
            predicate_coverage_logit=predicate["predicate_coverage_logit"],
            sufficiency_logit=sufficiency["sufficiency_logit"],
            entitlement_prob=product["entitlement_prob"],
            positive_energy=polarity["positive_energy"],
            negative_energy=polarity["negative_energy"],
            polarity_margin=polarity["polarity_margin"],
            polarity_strength=polarity["polarity_strength"],
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
            sufficiency_prob=sufficiency["sufficiency_prob"],
        )

        calibrated_logits = product["logits"] + composer["composer_correction_logits"]

        losses: dict[str, torch.Tensor | None] = {
            "final_loss": None,
            "product_final_loss": None,
            "label_loss": None,
            "product_label_loss": None,
            "frame_loss": None,
            "predicate_loss": None,
            "predicate_coverage_loss": None,
            "sufficiency_loss": None,
            "polarity_loss": None,
        }
        if final_labels is not None:
            labels = final_labels.long()
            losses["final_loss"] = F.cross_entropy(calibrated_logits, labels)
            losses["product_final_loss"] = F.cross_entropy(product["logits"], labels)
            losses["label_loss"] = losses["final_loss"]
            losses["product_label_loss"] = losses["product_final_loss"]
        if frame_compatible_labels is not None:
            losses["frame_loss"] = F.binary_cross_entropy_with_logits(
                frame["frame_logit"], frame_compatible_labels.float()
            )
        if predicate_covered_labels is not None:
            predicate_loss = F.binary_cross_entropy_with_logits(
                predicate["predicate_coverage_logit"],
                predicate_covered_labels.float(),
            )
            losses["predicate_loss"] = predicate_loss
            losses["predicate_coverage_loss"] = predicate_loss
        if sufficiency_labels is not None:
            losses["sufficiency_loss"] = F.binary_cross_entropy_with_logits(
                sufficiency["sufficiency_logit"], sufficiency_labels.float()
            )
        if polarity_labels is not None:
            if polarity_mask is None:
                polarity_mask = (
                    final_labels != 1
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
                    polarity_logits[active], polarity_labels[active].long()
                )

        active_losses: list[torch.Tensor] = []
        if losses["label_loss"] is not None:
            active_losses.append(losses["label_loss"])
        if losses["product_label_loss"] is not None:
            active_losses.append(self.product_loss_weight * losses["product_label_loss"])
        for key in ("frame_loss", "predicate_loss", "sufficiency_loss", "polarity_loss"):
            if losses[key] is not None:
                active_losses.append(losses[key])
        total_loss = torch.stack(active_losses).sum() if active_losses else None

        output = {
            **composer,
            "composer_logits": calibrated_logits,
            "logits": calibrated_logits,
            "predictions": calibrated_logits.argmax(dim=-1),
            "product_logits": product["logits"],
            "product_predictions": product["logits"].argmax(dim=-1),
            "product_entitlement_prob": product["entitlement_prob"],
            "product_support_logit": product["support_logit"],
            "product_refute_logit": product["refute_logit"],
            "product_not_entitled_logit": product["not_entitled_logit"],
            "entitlement_prob": product["entitlement_prob"],
            "support_logit": product["support_logit"],
            "refute_logit": product["refute_logit"],
            "not_entitled_logit": product["not_entitled_logit"],
            **frame,
            **predicate,
            **sufficiency,
            **polarity,
            "token_states": token_states if return_token_states else None,
            "loss": total_loss,
            "losses": losses,
            **losses,
        }
        return output
