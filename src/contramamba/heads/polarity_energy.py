from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PolarityEnergyHead(nn.Module):
    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        energy_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(
                frame_size + predicate_size + sufficiency_size, energy_size
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(energy_size),
        )
        self.positive_head = nn.Linear(energy_size, 1)
        self.negative_head = nn.Linear(energy_size, 1)

    def forward(
        self,
        frame_pair_repr: torch.Tensor,
        predicate_pair_repr: torch.Tensor,
        sufficiency_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        polarity_features = self.feature_projector(
            torch.cat(
                [frame_pair_repr, predicate_pair_repr, sufficiency_repr], dim=-1
            )
        )
        positive_energy = F.softplus(
            self.positive_head(polarity_features).squeeze(-1)
        )
        negative_energy = F.softplus(
            self.negative_head(polarity_features).squeeze(-1)
        )
        return {
            "positive_energy": positive_energy,
            "negative_energy": negative_energy,
            "polarity_margin": positive_energy - negative_energy,
            "polarity_strength": torch.sqrt(
                positive_energy.square() + negative_energy.square() + 1e-8
            ),
            "polarity_features": polarity_features,
        }

