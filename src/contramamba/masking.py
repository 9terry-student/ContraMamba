"""Mask validation and pooling shared by the v5 heads."""

from __future__ import annotations

import torch


def validate_pair_masks(
    token_states: torch.Tensor,
    attention_mask: torch.Tensor,
    claim_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if token_states.ndim != 3:
        raise ValueError("token_states must have shape [B, T, H]")

    expected_shape = token_states.shape[:2]
    masks = {
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
    }
    for name, mask in masks.items():
        if mask.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")

    attention = attention_mask.bool()
    claim = claim_mask.bool()
    evidence = evidence_mask.bool()

    if torch.any(claim & evidence):
        raise ValueError("claim_mask and evidence_mask must be disjoint")
    if torch.any(claim & ~attention) or torch.any(evidence & ~attention):
        raise ValueError("claim/evidence masks must be subsets of attention_mask")
    if torch.any(claim.sum(dim=1) == 0):
        raise ValueError("claim_mask must be non-empty for every example")
    if torch.any(evidence.sum(dim=1) == 0):
        raise ValueError("evidence_mask must be non-empty for every example")

    return attention, claim, evidence


def masked_pool(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=states.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (states * weights).sum(dim=1) / denominator

