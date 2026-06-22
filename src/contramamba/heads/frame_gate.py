from __future__ import annotations

import torch
from torch import nn

from ..masking import masked_pool, validate_pair_masks


class FrameGate(nn.Module):
    """Category-free, pair-level latent frame compatibility."""

    def __init__(
        self,
        hidden_size: int,
        frame_size: int,
        dropout: float = 0.1,
        return_token_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.return_token_diagnostics = return_token_diagnostics
        self.project = nn.Sequential(
            nn.Linear(hidden_size, frame_size),
            nn.GELU(),
            nn.LayerNorm(frame_size),
        )
        self.pair_projector = nn.Sequential(
            nn.Linear(frame_size * 4, frame_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(frame_size),
        )
        self.frame_classifier = nn.Linear(frame_size, 1)
        self.token_diagnostic = (
            nn.Linear(frame_size, 1) if return_token_diagnostics else None
        )

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        attention, claim, evidence = validate_pair_masks(
            token_states, attention_mask, claim_mask, evidence_mask
        )
        projected = self.project(token_states)
        claim_state = masked_pool(projected, claim)
        evidence_state = masked_pool(projected, evidence)
        pair_features = torch.cat(
            [
                claim_state,
                evidence_state,
                torch.abs(claim_state - evidence_state),
                claim_state * evidence_state,
            ],
            dim=-1,
        )
        pair_repr = self.pair_projector(pair_features)
        frame_logit = self.frame_classifier(pair_repr).squeeze(-1)

        token_scores = None
        if self.token_diagnostic is not None:
            token_scores = torch.sigmoid(self.token_diagnostic(projected).squeeze(-1))
            token_scores = token_scores * attention.to(token_scores.dtype)

        return {
            "frame_logit": frame_logit,
            "frame_prob": torch.sigmoid(frame_logit),
            "claim_frame_state": claim_state,
            "evidence_frame_state": evidence_state,
            "frame_pair_repr": pair_repr,
            "frame_token_scores": token_scores,
        }

