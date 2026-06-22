from __future__ import annotations

import torch
from torch import nn


class SufficiencyGate(nn.Module):
    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(frame_size + predicate_size + 2, sufficiency_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(sufficiency_size),
        )
        self.classifier = nn.Linear(sufficiency_size, 1)

    def forward(
        self,
        frame_pair_repr: torch.Tensor,
        predicate_pair_repr: torch.Tensor,
        frame_prob: torch.Tensor,
        predicate_coverage_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [
                frame_pair_repr,
                predicate_pair_repr,
                frame_prob.unsqueeze(-1),
                predicate_coverage_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        sufficiency_repr = self.projector(features)
        sufficiency_logit = self.classifier(sufficiency_repr).squeeze(-1)
        return {
            "sufficiency_logit": sufficiency_logit,
            "sufficiency_prob": torch.sigmoid(sufficiency_logit),
            "sufficiency_repr": sufficiency_repr,
        }

