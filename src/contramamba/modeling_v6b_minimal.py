"""ContraMamba-v6B-minimal: v5 base + learnable slot-level comparator alphas.

Minimal design: no composer, no dual loss paths. Just v5 with final logit modulation
for temporal and predicate mismatch signals. All CE/pairwise/intervention losses
consume the final modulated logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .heads import (
    FinalEntitlementDecisionHead,
    FrameGate,
    PolarityEnergyHead,
    PredicateCoverageHead,
    SufficiencyGate,
)


def _inverse_softplus(target: float) -> float:
    """Compute x such that softplus(x) ≈ target.

    softplus(x) = log(1 + exp(x))
    Solving: log(1 + exp(x)) = target
             exp(x) = exp(target) - 1
             x = log(exp(target) - 1)
    """
    if target <= 0:
        raise ValueError("target must be positive")
    return math.log(math.expm1(target))


class ContraMambaV6BMinimal(nn.Module):
    """Minimal v6B: v5 base + learnable temporal/predicate comparator alphas.

    Applies learnable slot-level logit modulation for temporal and predicate
    mismatch flags. No composer, no dual logit paths. Final logits are the only
    logits used for loss and prediction.
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
        use_temporal_comparator: bool = False,
        use_predicate_comparator: bool = False,
        alpha_temporal_init: float = 1.25,
        alpha_predicate_init: float = 1.25,
        use_boundary_head: bool = False,
        use_frame_violation_head: bool = False,
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
        self.use_temporal_comparator = use_temporal_comparator
        self.use_predicate_comparator = use_predicate_comparator

        # V5 heads (unchanged)
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
        self.decision_head = FinalEntitlementDecisionHead(decision_mode=decision_mode)

        # Learnable alphas for comparators (initialized near calibrated values)
        self.alpha_temporal_raw: nn.Parameter | None = None
        self.alpha_predicate_raw: nn.Parameter | None = None

        if use_temporal_comparator:
            raw_val = _inverse_softplus(alpha_temporal_init)
            self.alpha_temporal_raw = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

        if use_predicate_comparator:
            raw_val = _inverse_softplus(alpha_predicate_init)
            self.alpha_predicate_raw = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

        # Stage22-A: preservation boundary head — distinguishes preservation-like records
        # (none, paraphrase) from frame-mismatch records (location_swap, role_swap, etc.).
        # Input: concatenated frame_pair_repr + predicate_pair_repr + sufficiency_repr.
        # Does NOT touch output["logits"] or output["base_logits"].
        self.boundary_head: nn.Linear | None = None
        if use_boundary_head:
            self.boundary_head = nn.Linear(
                frame_size + predicate_size + sufficiency_size, 1
            )

        # Stage22-A3: frame violation head — distinguishes frame-violating interventions
        # (entity_swap, event_swap, location_swap, role_swap, title_name_swap; violation=1)
        # from non-violating interventions (none, paraphrase, sufficiency/polarity types; violation=0).
        # Same input as boundary_head. Does NOT touch output["logits"] or output["base_logits"].
        self.frame_violation_head: nn.Linear | None = None
        if use_frame_violation_head:
            self.frame_violation_head = nn.Linear(
                frame_size + predicate_size + sufficiency_size, 1
            )

    def alpha_temporal(self) -> float | torch.Tensor:
        """Return softplus-constrained temporal alpha."""
        if self.alpha_temporal_raw is None:
            return 0.0
        return F.softplus(self.alpha_temporal_raw)

    def alpha_predicate(self) -> float | torch.Tensor:
        """Return softplus-constrained predicate alpha."""
        if self.alpha_predicate_raw is None:
            return 0.0
        return F.softplus(self.alpha_predicate_raw)

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
    ) -> dict[str, Any]:
        """Forward pass with optional temporal/predicate logit modulation."""
        del intervention_types, pair_ids

        # Encode
        if encoder_hidden_states is None:
            backbone_outputs = self.mamba(input_ids=input_ids)
            token_states = backbone_outputs.last_hidden_state
        else:
            if encoder_hidden_states.shape[:2] != input_ids.shape:
                raise ValueError(
                    "encoder_hidden_states must match input_ids batch/sequence dimensions"
                )
            token_states = encoder_hidden_states

        # Slot gates (unchanged from V5)
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

        # Stage22-A / A3: shared slot concatenation for auxiliary diagnostic heads.
        # Computed once; reused by both boundary_head and frame_violation_head.
        # Does NOT modify final_logits or base_logits.
        _aux_input: torch.Tensor | None = None
        if self.boundary_head is not None or self.frame_violation_head is not None:
            _aux_input = torch.cat(
                [
                    frame["frame_pair_repr"],
                    predicate["predicate_pair_repr"],
                    sufficiency["sufficiency_repr"],
                ],
                dim=-1,
            )

        # Stage22-A: preservation boundary head (diagnostic/training signal only)
        # Output: boundary_logit [B], boundary_prob [B]; None when head is disabled.
        boundary_logit: torch.Tensor | None = None
        boundary_prob: torch.Tensor | None = None
        if self.boundary_head is not None and _aux_input is not None:
            boundary_logit = self.boundary_head(_aux_input).squeeze(-1)
            boundary_prob = torch.sigmoid(boundary_logit)

        # Stage22-A3: frame violation head (diagnostic/training signal only)
        # Output: frame_violation_logit [B], frame_violation_prob [B]; None when disabled.
        frame_violation_logit: torch.Tensor | None = None
        frame_violation_prob: torch.Tensor | None = None
        if self.frame_violation_head is not None and _aux_input is not None:
            frame_violation_logit = self.frame_violation_head(_aux_input).squeeze(-1)
            frame_violation_prob = torch.sigmoid(frame_violation_logit)

        # Base logits (V5 standard)
        decision = self.decision_head(
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
            sufficiency_prob=sufficiency["sufficiency_prob"],
            positive_energy=polarity["positive_energy"],
            negative_energy=polarity["negative_energy"],
            decision_mode=decision_mode,
        )
        base_logits = decision["logits"]

        # Apply final logit modulation
        final_logits = base_logits
        temporal_flag_count = 0
        predicate_flag_count = 0

        if self.use_temporal_comparator and temporal_mismatch_flags is not None:
            alpha = self.alpha_temporal()
            active = temporal_mismatch_flags.bool()
            if torch.any(active):
                final_logits = final_logits.clone()
                final_logits[active, 0] -= alpha  # SUPPORT
                final_logits[active, 1] += alpha  # NOT_ENTITLED
                final_logits[active, 2] -= alpha  # REFUTE
                temporal_flag_count = int(active.sum().item())

        if self.use_predicate_comparator and predicate_mismatch_flags is not None:
            alpha = self.alpha_predicate()
            active = predicate_mismatch_flags.bool()
            if torch.any(active):
                final_logits = final_logits.clone()
                final_logits[active, 0] -= alpha  # SUPPORT
                final_logits[active, 1] += alpha  # NOT_ENTITLED
                final_logits[active, 2] -= alpha  # REFUTE
                predicate_flag_count = int(active.sum().item())

        # Compute losses (using final_logits only)
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
                    polarity_logits[active], polarity_labels[active]
                )

        active_losses = [loss for loss in losses.values() if loss is not None]
        total_loss = torch.stack(active_losses).sum() if active_losses else None

        # Return: final_logits as the only logits for downstream losses
        return {
            **decision,
            "logits": final_logits,  # ← FINAL logits for all downstream loss paths
            "base_logits": base_logits,  # ← diagnostic only
            "predictions": final_logits.argmax(dim=-1),  # ← derived from final logits
            "alpha_temporal": self.alpha_temporal(),
            "alpha_predicate": self.alpha_predicate(),
            "temporal_flag_count": temporal_flag_count,
            "predicate_flag_count": predicate_flag_count,
            "final_logits_used": True,  # ← assertion for smoke test
            # Stage22-A: boundary head outputs (None when head is disabled)
            "boundary_logit": boundary_logit,
            "boundary_prob": boundary_prob,
            # Stage22-A3: frame violation head outputs (None when head is disabled)
            "frame_violation_logit": frame_violation_logit,
            "frame_violation_prob": frame_violation_prob,
            **frame,
            **predicate,
            **sufficiency,
            **polarity,
            "token_states": token_states if return_token_states else None,
            "loss": total_loss,
            **losses,
        }
