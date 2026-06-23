"""Helper functions to extract temporal and predicate flags from training records.

This module provides minimal flag extraction for Stage21. Flags indicate whether
a record exhibits temporal_mismatch or predicate_mismatch, so that comparator
alphas can be applied during training.

Constraints:
- Do NOT use time_swap for temporal detection (forbidden).
- Temporal flags should come from Stage17 external detector or stage15 probe.
- Predicate flags can use Stage20 lexical detector or stage15 probe.
"""

from __future__ import annotations

from typing import Sequence

import torch


def temporal_mismatch_flags_from_probe(
    records: Sequence[dict],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract temporal_mismatch flags from stage15_probe_type.

    Args:
        records: list of dicts, each with optional 'stage15_probe_type' key
        device: torch device for output tensor

    Returns:
        [batch] tensor of 0/1 flags, 1 if stage15_probe_type == "temporal_mismatch"
    """
    flags = [
        1 if record.get("stage15_probe_type") == "temporal_mismatch" else 0
        for record in records
    ]
    return torch.tensor(flags, dtype=torch.long, device=device)


def predicate_mismatch_flags_from_probe(
    records: Sequence[dict],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract predicate_mismatch flags from stage15_probe_type.

    Args:
        records: list of dicts, each with optional 'stage15_probe_type' key
        device: torch device for output tensor

    Returns:
        [batch] tensor of 0/1 flags, 1 if stage15_probe_type == "predicate_mismatch"
    """
    flags = [
        1 if record.get("stage15_probe_type") == "predicate_mismatch" else 0
        for record in records
    ]
    return torch.tensor(flags, dtype=torch.long, device=device)


def predicate_mismatch_flags_from_intervention_type(
    records: Sequence[dict],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Heuristic: extract predicate_mismatch flags from intervention_type.

    This is a fallback for controlled_v5 data, which does not have stage15_probe_type.
    Uses intervention_type == "predicate_swap" as a proxy.

    WARNING: This is a heuristic approximation, not a ground-truth detector.
    Use only for smoke testing or controlled training when stage15 probe is unavailable.

    Args:
        records: list of dicts, each with 'intervention_type' key
        device: torch device for output tensor

    Returns:
        [batch] tensor of 0/1 flags, 1 if intervention_type == "predicate_swap"
    """
    flags = [
        1 if record.get("intervention_type") == "predicate_swap" else 0
        for record in records
    ]
    return torch.tensor(flags, dtype=torch.long, device=device)


def temporal_mismatch_flags_none(
    records: Sequence[dict],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return zeros (no temporal detection) as a safe default for controlled data.

    Temporal flags should ideally come from Stage17 external temporal comparator
    or stage15 probe. Since we cannot use time_swap from controlled_v5, and there
    is no Stage17 detector integrated into training, this returns all-zeros.

    Args:
        records: list of dicts (unused)
        device: torch device for output tensor

    Returns:
        [batch] tensor of all zeros
    """
    return torch.zeros(len(records), dtype=torch.long, device=device)
