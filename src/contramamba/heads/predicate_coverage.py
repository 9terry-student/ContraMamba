from __future__ import annotations

import torch
from torch import nn

from ..masking import masked_pool, validate_pair_masks


class PredicateCoverageHead(nn.Module):
    """Pair-level estimate that evidence covers the claim predicate."""

    def __init__(
        self,
        hidden_size: int,
        frame_size: int,
        predicate_size: int,
        dropout: float = 0.1,
        return_token_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_size, predicate_size),
            nn.GELU(),
            nn.LayerNorm(predicate_size),
        )
        pair_input_size = predicate_size * 4 + frame_size + 1
        self.pair_projector = nn.Sequential(
            nn.Linear(pair_input_size, predicate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(predicate_size),
        )
        self.coverage_classifier = nn.Linear(predicate_size, 1)
        self.token_diagnostic = (
            nn.Linear(predicate_size, 1) if return_token_diagnostics else None
        )

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        claim_frame_state: torch.Tensor,
        evidence_frame_state: torch.Tensor,
        frame_pair_repr: torch.Tensor,
        frame_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        del claim_frame_state, evidence_frame_state  # Reserved for later ablations.
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
                frame_pair_repr,
                frame_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_repr = self.pair_projector(pair_features)
        coverage_logit = self.coverage_classifier(pair_repr).squeeze(-1)

        token_scores = None
        if self.token_diagnostic is not None:
            token_scores = torch.sigmoid(self.token_diagnostic(projected).squeeze(-1))
            token_scores = token_scores * attention.to(token_scores.dtype)

        return {
            "predicate_coverage_logit": coverage_logit,
            "predicate_coverage_prob": torch.sigmoid(coverage_logit),
            "claim_predicate_state": claim_state,
            "evidence_predicate_state": evidence_state,
            "predicate_pair_repr": pair_repr,
            "predicate_token_scores": token_scores,
        }

