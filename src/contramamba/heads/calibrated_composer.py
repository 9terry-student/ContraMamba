from __future__ import annotations

import torch
from torch import nn


class CalibratedComposer(nn.Module):
    """Residual calibrated composer for ContraMamba-v6A.

    Produces correction logits. Final calibrated logits are formed in the model as:
        final_logits = product_logits + correction_logits

    Label order:
        0 = REFUTE
        1 = NOT_ENTITLED
        2 = SUPPORT
    """

    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.1,
        input_size: int = 10,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.feature_norm = nn.Identity()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.output = nn.Linear(hidden_size, 3)

        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        frame_logit: torch.Tensor,
        predicate_coverage_logit: torch.Tensor,
        sufficiency_logit: torch.Tensor,
        entitlement_prob: torch.Tensor,
        positive_energy: torch.Tensor,
        negative_energy: torch.Tensor,
        polarity_margin: torch.Tensor,
        polarity_strength: torch.Tensor,
        frame_prob: torch.Tensor,
        predicate_coverage_prob: torch.Tensor,
        sufficiency_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gate_stack = torch.stack(
            [frame_prob, predicate_coverage_prob, sufficiency_prob],
            dim=-1,
        )
        min_gate = gate_stack.min(dim=-1).values
        max_gate = gate_stack.max(dim=-1).values
        gate_range = max_gate - min_gate

        features = torch.stack(
            [
                frame_logit,
                predicate_coverage_logit,
                sufficiency_logit,
                entitlement_prob,
                positive_energy,
                negative_energy,
                polarity_margin,
                polarity_strength,
                min_gate,
                gate_range,
            ],
            dim=-1,
        )

        features = self.feature_norm(features)
        hidden = self.hidden(features)
        correction_logits = self.output(hidden)

        return {
            "composer_features": features,
            "composer_hidden": hidden,
            "composer_correction_logits": correction_logits,
            "composer_refute_logit": correction_logits[..., 0],
            "composer_not_entitled_logit": correction_logits[..., 1],
            "composer_support_logit": correction_logits[..., 2],
        }
